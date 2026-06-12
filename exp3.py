import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from src.data import load_amazon_spanish, preprocess_corpus
from src.word2vec_model import Vocabulary, Word2Vec1Layer
from src.trainer import train_word2vec
from src.classifier import build_document_matrix, train_classifier, evaluate

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MAX_TRAIN = 40000
BATCH     = 512
DEVICE    = "cuda"
EPOCHS    = 5


# --- TF-IDF baseline ---

def run_tfidf(train_texts, train_labels, test_texts, test_labels):
    print("\n--- TF-IDF + Logistic Regression ---")
    # Texto plano (sin tokenizar, TfidfVectorizer lo hace internamente)
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
        max_features=50000,
    )
    X_train = vec.fit_transform(train_texts)
    X_test  = vec.transform(test_texts)

    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(X_train, train_labels)

    y_pred   = clf.predict(X_test)
    acc      = accuracy_score(test_labels, y_pred)
    f1_macro = f1_score(test_labels, y_pred, average="macro")

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-macro : {f1_macro:.4f}")
    print(f"  Features : {X_train.shape[1]:,}")
    return {"accuracy": acc, "f1_macro": f1_macro, "n_features": X_train.shape[1]}


# --- Ablacion de dimension de embeddings ---

def run_dim_ablation(train_tokens, train_labels, test_tokens, test_labels, vocab):
    dims    = [25, 50, 100, 150, 200]
    results = []

    print("\n--- Ablacion: dimension de embeddings (Adam, 5 epochs) ---")
    for d in dims:
        print(f"\n  d={d}")
        model = Word2Vec1Layer(vocab.vocab_size, embed_dim=d)
        res   = train_word2vec(
            model, train_tokens, vocab,
            optimizer_name="Adam", learning_rate=0.001,
            epochs=EPOCHS, batch_size=BATCH,
            window_size=5, num_negatives=5, device=DEVICE
        )
        emb     = model.get_embeddings()
        X_train = build_document_matrix(train_tokens, vocab, emb)
        X_test  = build_document_matrix(test_tokens,  vocab, emb)
        clf     = train_classifier(X_train, train_labels)
        metrics = evaluate(clf, X_test, test_labels, f"Word2Vec d={d}")
        results.append({
            "dim":      d,
            "f1_macro": metrics["f1_macro"],
            "accuracy": metrics["accuracy"],
            "loss_final": res["loss_per_epoch"][-1],
        })
        print(f"  d={d}  F1={metrics['f1_macro']:.4f}  "
              f"loss={res['loss_per_epoch'][-1]:.4f}")

    return results


# --- Figuras ---

def plot_results(tfidf_res, dim_results, w2v_best_f1):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel izquierdo: comparacion TF-IDF vs Word2Vec
    ax = axes[0]
    modelos = ["TF-IDF\n+ LogReg", "Word2Vec\n+ LogReg\n(mejor)"]
    f1s     = [tfidf_res["f1_macro"], w2v_best_f1]
    colors  = ["#95A5A6", "#185FA5"]
    bars    = ax.bar(modelos, f1s, color=colors, width=0.5, edgecolor="white")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("F1-macro")
    ax.set_title("TF-IDF vs Word2Vec")
    ax.grid(True, alpha=0.3, linestyle=":", axis="y")
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Panel derecho: F1 vs dimension de embeddings
    ax2   = axes[1]
    dims  = [r["dim"]      for r in dim_results]
    f1s_d = [r["f1_macro"] for r in dim_results]
    ax2.plot(dims, f1s_d, color="#185FA5", linewidth=2.5,
             marker="o", markersize=8)
    ax2.axhline(y=tfidf_res["f1_macro"], color="#95A5A6",
                linewidth=1.5, linestyle="--", label="TF-IDF baseline")
    ax2.set_title("F1-macro vs Dimension de embeddings")
    ax2.set_xlabel("Dimension (d)")
    ax2.set_ylabel("F1-macro")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Anotar el codo
    best_idx = int(np.argmax(f1s_d))
    ax2.annotate(
        f"codo: d={dims[best_idx]}",
        xy=(dims[best_idx], f1s_d[best_idx]),
        xytext=(dims[best_idx] + 15, f1s_d[best_idx] - 0.01),
        fontsize=9, color="#E24B4A",
        arrowprops=dict(arrowstyle="->", color="#E24B4A"),
    )

    fig.suptitle("Experimento 3: TF-IDF baseline + ablacion de dimension",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    fname = RESULTS_DIR / "exp3_tfidf_ablacion.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {fname}")


def main():
    print("Cargando datos...")
    (train_texts, train_labels, _, _,
     test_texts,  test_labels) = load_amazon_spanish(max_train=MAX_TRAIN)

    train_tokens = preprocess_corpus(train_texts)
    test_tokens  = preprocess_corpus(test_texts)
    vocab = Vocabulary(min_count=5).build(train_tokens)

    # TF-IDF
    tfidf_res = run_tfidf(train_texts, train_labels, test_texts, test_labels)

    # Ablacion de dimension
    dim_results = run_dim_ablation(
        train_tokens, train_labels, test_tokens, test_labels, vocab
    )

    # F1 del mejor Word2Vec del experimento principal (RMSProp, 1 capa)
    w2v_best_f1 = 0.8942

    # Figura
    print("\nGenerando figura...")
    plot_results(tfidf_res, dim_results, w2v_best_f1)

    # Resumen
    print("\nResumen TF-IDF:")
    print(f"  F1-macro : {tfidf_res['f1_macro']:.4f}")
    print(f"  Accuracy : {tfidf_res['accuracy']:.4f}")
    print("\nResumen ablacion de dimension:")
    print(f"  {'d':<6} {'F1-macro':>9} {'Loss final':>11}")
    print("  " + "-"*28)
    for r in dim_results:
        print(f"  {r['dim']:<6} {r['f1_macro']:>9.4f} {r['loss_final']:>11.4f}")

    with open(RESULTS_DIR / "exp3_resultados.pkl", "wb") as f:
        pickle.dump({"tfidf": tfidf_res, "dim_ablation": dim_results}, f)

    print("\nExperimento 3 completo.")


if __name__ == "__main__":
    main()