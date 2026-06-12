import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from src.data import load_amazon_spanish, preprocess_corpus
from src.word2vec_model import Vocabulary, Word2Vec1Layer
from src.trainer import train_word2vec

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MAX_TRAIN = 40000
EMBED_DIM = 100
BATCH     = 512
DEVICE    = "cuda"
EPOCHS    = 5

# Palabras clave para analizar similitud coseno
QUERY_WORDS = [
    "bueno", "malo", "excelente", "horrible", "calidad",
    "producto", "envio", "precio", "rapido", "lento",
    "recomiendo", "devolucion", "roto", "perfecto", "pesimo"
]

# Palabras de sentimiento para colorear t-SNE
POSITIVE_WORDS = {
    "bueno", "excelente", "perfecto", "genial", "increible",
    "maravilloso", "recomiendo", "feliz", "rapido", "facil",
    "bonito", "util", "calidad", "satisfecho", "super"
}
NEGATIVE_WORDS = {
    "malo", "horrible", "pesimo", "terrible", "roto",
    "devolucion", "lento", "caro", "decepcion", "fraude",
    "basura", "defecto", "problema", "queja", "dano"
}


def get_similar_words(word, vocab, embeddings, top_n=5):
    if word not in vocab.word2idx:
        return []
    idx  = vocab.word2idx[word]
    vec  = embeddings[idx].reshape(1, -1)
    sims = cosine_similarity(vec, embeddings)[0]
    sims[idx] = -1  # excluir la palabra misma
    top_idx = np.argsort(sims)[::-1][:top_n]
    return [(vocab.idx2word[i], float(sims[i])) for i in top_idx]


def print_similarity_table(vocab, embeddings):
    print("\nSimilitud coseno -- Top 5 palabras similares:")
    print("-" * 55)
    found = 0
    for word in QUERY_WORDS:
        similares = get_similar_words(word, vocab, embeddings, top_n=5)
        if similares:
            similares_str = ", ".join(
                f"{w}({s:.2f})" for w, s in similares
            )
            print(f"  {word:<15} -> {similares_str}")
            found += 1
    print(f"\n  Palabras encontradas en vocabulario: {found}/{len(QUERY_WORDS)}")
    return found


def plot_tsne_colored(vocab, embeddings, n_words=400):
    """
    t-SNE coloreado:
      azul   = palabras positivas conocidas
      rojo   = palabras negativas conocidas
      gris   = resto
    """
    n_words = min(n_words, vocab.vocab_size)
    words   = [vocab.idx2word[i] for i in range(n_words)]
    emb_sub = embeddings[:n_words]

    print(f"\nCalculando t-SNE para {n_words} palabras...")
    tsne   = TSNE(n_components=2, random_state=42,
                  perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(emb_sub)

    colors = []
    sizes  = []
    for w in words:
        if w in POSITIVE_WORDS:
            colors.append("#185FA5")   # azul
            sizes.append(60)
        elif w in NEGATIVE_WORDS:
            colors.append("#E24B4A")   # rojo
            sizes.append(60)
        else:
            colors.append("#CCCCCC")   # gris
            sizes.append(15)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(coords[:, 0], coords[:, 1],
               c=colors, s=sizes, alpha=0.7, linewidths=0)

    # Etiquetar solo palabras positivas y negativas
    for i, word in enumerate(words):
        if word in POSITIVE_WORDS or word in NEGATIVE_WORDS:
            color = "#0A3D6B" if word in POSITIVE_WORDS else "#8B1A1A"
            ax.annotate(word, coords[i],
                        fontsize=8, fontweight="bold",
                        color=color, alpha=0.9)

    # Leyenda manual
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#185FA5", markersize=9,
               label="Palabras positivas"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#E24B4A", markersize=9,
               label="Palabras negativas"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#CCCCCC", markersize=9,
               label="Resto del vocabulario"),
    ]
    ax.legend(handles=legend, fontsize=10, loc="upper right")
    ax.set_title(
        "Embeddings Word2Vec -- t-SNE coloreado por sentimiento\n"
        "(azul=positivo, rojo=negativo)",
        fontsize=13
    )
    ax.axis("off")
    fig.tight_layout()

    fname = RESULTS_DIR / "exp4_tsne_colored.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {fname}")


def plot_similarity_heatmap(vocab, embeddings):
    """
    Heatmap de similitud coseno entre las palabras clave
    que existen en el vocabulario.
    """
    available = [w for w in QUERY_WORDS if w in vocab.word2idx]
    if len(available) < 3:
        print("Pocas palabras clave en vocabulario, saltando heatmap.")
        return

    indices = [vocab.word2idx[w] for w in available]
    vecs    = embeddings[indices]
    sim_mat = cosine_similarity(vecs)

    fig, ax = plt.subplots(figsize=(len(available) * 0.7 + 2,
                                    len(available) * 0.7 + 1))
    im = ax.imshow(sim_mat, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Similitud coseno")

    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(available, fontsize=9)

    for i in range(len(available)):
        for j in range(len(available)):
            ax.text(j, i, f"{sim_mat[i,j]:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="white" if sim_mat[i, j] > 0.6 else "black")

    ax.set_title("Similitud coseno entre palabras clave del corpus",
                 fontsize=12)
    fig.tight_layout()

    fname = RESULTS_DIR / "exp4_similarity_heatmap.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {fname}")


def main():
    print("Cargando datos...")
    (train_texts, train_labels, _, _,
     test_texts,  test_labels) = load_amazon_spanish(max_train=MAX_TRAIN)

    train_tokens = preprocess_corpus(train_texts)
    vocab = Vocabulary(min_count=5).build(train_tokens)

    print("\nEntrenando Word2Vec (Adam, 5 epochs)...")
    model = Word2Vec1Layer(vocab.vocab_size, embed_dim=EMBED_DIM)
    res   = train_word2vec(
        model, train_tokens, vocab,
        optimizer_name="Adam", learning_rate=0.001,
        epochs=EPOCHS, batch_size=BATCH,
        window_size=5, num_negatives=5, device=DEVICE
    )
    embeddings = model.get_embeddings()

    print("\n" + "="*55)
    print("  Analisis de similitud coseno")
    print("="*55)
    print_similarity_table(vocab, embeddings)

    print("\n" + "="*55)
    print("  Generando figuras")
    print("="*55)
    plot_tsne_colored(vocab, embeddings, n_words=400)
    plot_similarity_heatmap(vocab, embeddings)

    with open(RESULTS_DIR / "exp4_resultados.pkl", "wb") as f:
        pickle.dump({
            "loss_per_epoch": res["loss_per_epoch"],
            "vocab_size": vocab.vocab_size,
        }, f)

    print("\nExperimento 4 completo.")
    print("Figuras generadas:")
    print("  results/exp4_tsne_colored.png")
    print("  results/exp4_similarity_heatmap.png")


if __name__ == "__main__":
    main()