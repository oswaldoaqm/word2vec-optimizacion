# exp6.py
# ============================================================
#  Experimento 6 — Duelo de embeddings:
#    Nuestro Word2Vec (entrenado in-domain en 40k reseñas de Amazon)
#    vs. SBWC preentrenado (Spanish Billion Word Corpus, ~1.5B palabras)
#
#  Idea: los 5 experimentos previos afinaron NUESTRO mejor Word2Vec.
#  Aqui lo ponemos a prueba contra un modelo "super entrenado" para
#  medir, sobre datasets realistas, cuanto importa el tamano del
#  corpus de entrenamiento y la transferibilidad fuera de dominio.
#
#  Diseno (comparacion justa): MISMO clasificador (Regresion Logistica)
#  y MISMO preprocesamiento en todos los casos; solo cambia la matriz de
#  embeddings. Para cada representacion se reportan TRES metricas, todas
#  evaluadas sobre el mismo conjunto de test:
#    1. Amazon (in-domain)     : clasificador entrenado en Amazon-train.
#    2. Tweets (transferencia) : clasificador de Amazon probado en tweets
#                                (zero-shot, sin re-entrenar).
#    3. Tweets (in-domain)     : clasificador entrenado en un split de
#                                tweets y probado en el test de tweets.
#  Ademas se reporta la COBERTURA de vocabulario (OOV) en los tweets para
#  cuantificar la limitacion de "vocabulario cerrado". TF-IDF se incluye
#  como referencia.
#
#  Dataset out-of-domain: cardiffnlp/tweet_sentiment_multilingual (split es),
#  benchmark XLM-T (Barbieri et al., LREC 2022). Se cargan los jsonl
#  directamente (sin scripts de la libreria datasets).
#
#  Requisitos:
#    pip install gensim
#    Descargar los embeddings SBWC (formato word2vec binario) desde:
#       https://crscardellino.github.io/SBWCE/
#    y colocar el archivo en la ruta SBWC_PATH (abajo).
#    Recomendado: SBW-vectors-300-min5.bin.gz  (300d, ~3GB)
#
#  Uso:  python exp6.py
# ============================================================

import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.data import load_amazon_spanish, preprocess_corpus
from src.word2vec_model import Vocabulary, Word2Vec1Layer
from src.trainer import train_word2vec
from src.classifier import train_classifier
from src.utils import set_seed

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Configuracion ────────────────────────────────────────────
MAX_TRAIN = 40000
OUR_DIM   = 300        # igualamos la dimension de SBWC (300d) para un duelo limpio
EPOCHS    = 5
BATCH     = 512
DEVICE    = "cuda"

# Embeddings SBWC preentrenados (formato word2vec binario .bin / .bin.gz)
# Descargar de https://crscardellino.github.io/SBWCE/
SBWC_PATH = "embeddings/SBW-vectors-300-min5.bin.gz"

# Dataset out-of-domain: tweets de sentimiento en español (CardiffNLP).
OOD_NAME    = "Tweets ES"
TWEETS_BASE = ("https://huggingface.co/datasets/cardiffnlp/"
               "tweet_sentiment_multilingual/resolve/main/data/spanish")
TWEETS_SPLITS = ["train", "validation", "test"]
TWEETS_DIR  = Path("data/tweets_es")


# ── Carga del dataset out-of-domain (tweets ES) ──────────────

def load_tweets_es():
    """
    Tweets de sentimiento en español (cardiffnlp/tweet_sentiment_multilingual).
    Etiquetas originales: 0=negative, 1=neutral, 2=positive.
    Binariza igual que Amazon: 0 -> negativo (0), 2 -> positivo (1), 1 descartado.
    Combina train+validation+test (luego se reparte en train/test para la
    evaluacion in-domain). Devuelve (textos, etiquetas).
    """
    print(f"Cargando {OOD_NAME} (cardiffnlp/tweet_sentiment_multilingual)...")
    TWEETS_DIR.mkdir(parents=True, exist_ok=True)

    texts, labels = [], []
    for split in TWEETS_SPLITS:
        local = TWEETS_DIR / f"{split}.jsonl"
        if not local.exists():
            url = f"{TWEETS_BASE}/{split}.jsonl"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    local.write_bytes(r.read())
            except Exception as e:
                raise RuntimeError(
                    f"\nNo se pudo descargar {url} ({e}).\n"
                    f"Descargalo manualmente con:\n"
                    f"  curl -L -A 'Mozilla/5.0' -o {local} {url}\n"
                    f"y vuelve a ejecutar exp6.py."
                )

        for line in local.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("tweet") or obj.get("sentence")
            lab  = obj.get("label", obj.get("target"))
            if text is None or lab is None:
                continue
            lab = int(lab)
            if lab == 0:
                labels.append(0)      # negative
            elif lab == 2:
                labels.append(1)      # positive
            else:
                continue              # neutral -> descartado
            texts.append(text)

    neg, pos = labels.count(0), labels.count(1)
    print(f"  {OOD_NAME}: {len(texts):,} tweets binarios (neg={neg:,} pos={pos:,})")
    return texts, labels


# ── Helpers de representacion ────────────────────────────────

def doc_vector_generic(tokens, get_vec, dim):
    """Promedia los vectores de las palabras cubiertas por la fuente."""
    vecs = [get_vec(t) for t in tokens]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def build_matrix(token_lists, get_vec, dim):
    return np.vstack([doc_vector_generic(t, get_vec, dim) for t in token_lists])


def coverage(token_lists, contains):
    """Fraccion de tokens (con repeticion) cubiertos por el vocabulario."""
    total = covered = 0
    for toks in token_lists:
        for t in toks:
            total += 1
            covered += 1 if contains(t) else 0
    return covered / max(total, 1)


def eval_rep(name, get_vec, dim,
             am_train_tokens, y_am_train, am_test_tokens, y_am_test,
             ood_train_tokens, y_ood_train, ood_test_tokens, y_ood_test):
    """
    Tres metricas (F1-macro), todas sobre conjuntos de test:
      - Amazon in-domain        (clf entrenado en Amazon-train)
      - Tweets transferencia    (mismo clf de Amazon, probado en tweets-test)
      - Tweets in-domain        (clf entrenado en tweets-train)
    """
    X_ood_test = build_matrix(ood_test_tokens, get_vec, dim)

    clf_am = train_classifier(build_matrix(am_train_tokens, get_vec, dim), y_am_train)
    f1_amazon   = f1_score(y_am_test,
                           clf_am.predict(build_matrix(am_test_tokens, get_vec, dim)),
                           average="macro")
    f1_transfer = f1_score(y_ood_test, clf_am.predict(X_ood_test), average="macro")

    clf_ood = train_classifier(build_matrix(ood_train_tokens, get_vec, dim), y_ood_train)
    f1_ood_indomain = f1_score(y_ood_test, clf_ood.predict(X_ood_test), average="macro")

    print(f"  {name:<30} Amazon={f1_amazon:.4f}  "
          f"{OOD_NAME}-transf={f1_transfer:.4f}  {OOD_NAME}-indom={f1_ood_indomain:.4f}")
    return {"name": name, "f1_amazon": f1_amazon,
            "f1_ood_transfer": f1_transfer, "f1_ood_indomain": f1_ood_indomain}


def eval_tfidf(am_train_texts, y_am_train, am_test_texts, y_am_test,
               ood_train_texts, y_ood_train, ood_test_texts, y_ood_test):
    """TF-IDF de referencia, con las mismas tres metricas."""
    def fit_eval(train_texts, y_train):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                              sublinear_tf=True, norm="l2", max_features=50000)
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(vec.fit_transform(train_texts), y_train)
        return vec, clf

    vec_am, clf_am = fit_eval(am_train_texts, y_am_train)
    f1_amazon   = f1_score(y_am_test, clf_am.predict(vec_am.transform(am_test_texts)), average="macro")
    f1_transfer = f1_score(y_ood_test, clf_am.predict(vec_am.transform(ood_test_texts)), average="macro")

    vec_ood, clf_ood = fit_eval(ood_train_texts, y_ood_train)
    f1_ood_indomain = f1_score(y_ood_test, clf_ood.predict(vec_ood.transform(ood_test_texts)), average="macro")

    print(f"  {'TF-IDF (referencia)':<30} Amazon={f1_amazon:.4f}  "
          f"{OOD_NAME}-transf={f1_transfer:.4f}  {OOD_NAME}-indom={f1_ood_indomain:.4f}")
    return {"name": "TF-IDF (referencia)", "f1_amazon": f1_amazon,
            "f1_ood_transfer": f1_transfer, "f1_ood_indomain": f1_ood_indomain}


# ── Figura ───────────────────────────────────────────────────

def plot_duel(results, cov_ours, cov_sbwc):
    names = [r["name"] for r in results]
    f1_am  = [r["f1_amazon"] for r in results]
    f1_tr  = [r["f1_ood_transfer"] for r in results]
    f1_id  = [r["f1_ood_indomain"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    x = np.arange(len(names))
    w = 0.27
    ax.bar(x - w, f1_am, w, label="Amazon (in-domain)", color="#185FA5")
    ax.bar(x,     f1_tr, w, label=f"{OOD_NAME} (transferencia)", color="#E2A04B")
    ax.bar(x + w, f1_id, w, label=f"{OOD_NAME} (in-domain)", color="#4BA36B")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Rendimiento por dominio y modo de evaluacion")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3, linestyle=":", axis="y")
    for xi in range(len(names)):
        for off, vals in [(-w, f1_am), (0, f1_tr), (w, f1_id)]:
            ax.text(xi + off, vals[xi] + 0.005, f"{vals[xi]:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax2 = axes[1]
    ax2.bar(["Nuestro W2V\n(40k Amazon)", "SBWC\n(1.5B palabras)"],
            [cov_ours, cov_sbwc], color=["#185FA5", "#4BA36B"], width=0.5)
    ax2.set_ylabel(f"Cobertura de tokens en {OOD_NAME}")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Vocabulario: cobertura fuera de dominio")
    ax2.grid(True, alpha=0.3, linestyle=":", axis="y")
    for xi, c in enumerate([cov_ours, cov_sbwc]):
        ax2.text(xi, c + 0.01, f"{c:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Experimento 6: Nuestro Word2Vec vs SBWC preentrenado", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = RESULTS_DIR / "exp6_duelo_sbwc.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {fname}")


# ── Main ─────────────────────────────────────────────────────

def main():
    set_seed(42)

    # 1. Amazon (in-domain): train + test
    print("Cargando Amazon...")
    (am_train_texts, am_train_labels, _, _,
     am_test_texts,  am_test_labels) = load_amazon_spanish(max_train=MAX_TRAIN)
    am_train_tokens = preprocess_corpus(am_train_texts)
    am_test_tokens  = preprocess_corpus(am_test_texts)
    vocab = Vocabulary(min_count=5).build(am_train_tokens)

    # 2. Entrenar NUESTRO mejor Word2Vec (Adam, d=300) en Amazon-train
    print(f"\nEntrenando nuestro Word2Vec (Adam, d={OUR_DIM}) en Amazon...")
    model = Word2Vec1Layer(vocab.vocab_size, embed_dim=OUR_DIM)
    train_word2vec(model, am_train_tokens, vocab,
                   optimizer_name="Adam", learning_rate=0.001,
                   epochs=EPOCHS, batch_size=BATCH,
                   window_size=5, num_negatives=5, device=DEVICE)
    emb_ours = model.get_embeddings()

    # 3. Cargar SBWC preentrenado
    if not Path(SBWC_PATH).exists():
        raise FileNotFoundError(
            f"\nNo se encontro el archivo SBWC en: {SBWC_PATH}\n"
            "Descargalo de https://crscardellino.github.io/SBWCE/ "
            "(recomendado: SBW-vectors-300-min5.bin.gz) y colocalo en esa ruta,\n"
            "o ajusta la constante SBWC_PATH al inicio de exp6.py."
        )
    print(f"\nCargando SBWC preentrenado desde {SBWC_PATH} ...")
    from gensim.models import KeyedVectors
    binary = SBWC_PATH.endswith(".bin") or SBWC_PATH.endswith(".bin.gz")
    kv = KeyedVectors.load_word2vec_format(SBWC_PATH, binary=binary)
    print(f"  SBWC cargado: {len(kv):,} palabras, dim={kv.vector_size}")

    # 4. Dataset out-of-domain (tweets ES) con split train/test propio
    ood_texts, ood_labels = load_tweets_es()
    ood_tr_texts, ood_te_texts, ood_tr_labels, ood_te_labels = train_test_split(
        ood_texts, ood_labels, test_size=0.30, random_state=42, stratify=ood_labels)
    ood_tr_tokens = preprocess_corpus(ood_tr_texts)
    ood_te_tokens = preprocess_corpus(ood_te_texts)
    print(f"  Split tweets: {len(ood_tr_texts):,} train / {len(ood_te_texts):,} test")

    # 5. Funciones de acceso a vectores
    def get_vec_ours(w):
        idx = vocab.word2idx.get(w)
        return emb_ours[idx] if idx is not None else None

    def get_vec_sbwc(w):
        return kv[w] if w in kv.key_to_index else None

    # 6. Cobertura de vocabulario en los tweets (test set)
    cov_ours = coverage(ood_te_tokens, lambda w: w in vocab.word2idx)
    cov_sbwc = coverage(ood_te_tokens, lambda w: w in kv.key_to_index)
    print(f"\nCobertura de tokens en {OOD_NAME}:")
    print(f"  Nuestro W2V (40k Amazon) : {cov_ours:.1%}")
    print(f"  SBWC (1.5B palabras)     : {cov_sbwc:.1%}")

    # 7. Evaluacion
    print(f"\n{'='*72}\nRESULTADOS (F1-macro)\n{'='*72}")
    results = []
    results.append(eval_rep(f"Nuestro W2V (Amazon, d={OUR_DIM})", get_vec_ours, OUR_DIM,
                            am_train_tokens, am_train_labels, am_test_tokens, am_test_labels,
                            ood_tr_tokens, ood_tr_labels, ood_te_tokens, ood_te_labels))
    results.append(eval_rep(f"SBWC preentrenado (d={kv.vector_size})", get_vec_sbwc, kv.vector_size,
                            am_train_tokens, am_train_labels, am_test_tokens, am_test_labels,
                            ood_tr_tokens, ood_tr_labels, ood_te_tokens, ood_te_labels))
    results.append(eval_tfidf(am_train_texts, am_train_labels, am_test_texts, am_test_labels,
                              ood_tr_texts, ood_tr_labels, ood_te_texts, ood_te_labels))

    # 8. Figura + guardado
    plot_duel(results, cov_ours, cov_sbwc)
    with open(RESULTS_DIR / "exp6_resultados.pkl", "wb") as f:
        pickle.dump({
            "results": results,
            "ood_name": OOD_NAME,
            "coverage_ours": cov_ours,
            "coverage_sbwc": cov_sbwc,
            "our_dim": OUR_DIM,
            "sbwc_dim": kv.vector_size,
            "n_ood_train": len(ood_tr_labels),
            "n_ood_test": len(ood_te_labels),
        }, f)

    print("\nExperimento 6 completo.")


if __name__ == "__main__":
    main()
