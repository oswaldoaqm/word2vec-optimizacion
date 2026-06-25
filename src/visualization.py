# src/visualization.py
# ============================================================
#  Visualizaciones del proyecto
#
#  1. Curvas de convergencia (pérdida por epoch, por optimizador)
#  2. Visualización t-SNE de embeddings
#  3. Tabla comparativa de resultados
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Paleta de colores consistente para los tres optimizadores
COLORS = {
    "SGD":     "#E24B4A",   # rojo
    "RMSProp": "#BA7517",   # ámbar
    "Adam":    "#185FA5",   # azul
}
LINE_STYLES = {
    "SGD":     "--",
    "RMSProp": "-.",
    "Adam":    "-",
}


# ── 1. Curvas de convergencia ────────────────────────────────

def plot_convergence_curves(results: dict, arch_name: str, save: bool = True):
    """
    Grafica las curvas de pérdida por epoch para cada optimizador.

    Parámetros
    ----------
    results   : dict  { "SGD": {...}, "RMSProp": {...}, "Adam": {...} }
                      Cada valor es el dict retornado por train_word2vec()
    arch_name : str   Nombre de la arquitectura ("1 capa" o "2 capas")
    save      : bool  Guarda la figura en results/
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for opt_name, res in results.items():
        losses = res["loss_per_epoch"]
        epochs = range(1, len(losses) + 1)
        ax.plot(
            epochs, losses,
            color=COLORS[opt_name],
            linestyle=LINE_STYLES[opt_name],
            linewidth=2,
            marker="o", markersize=5,
            label=opt_name
        )

    ax.set_title(f"Curvas de convergencia — Word2Vec {arch_name}", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Pérdida (Negative Sampling)", fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()

    if save:
        fname = RESULTS_DIR / f"convergencia_{arch_name.replace(' ', '_')}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {fname}")

    plt.close()
    return fig


def plot_convergence_comparison(results_1layer: dict, results_2layer: dict,
                                 save: bool = True):
    """
    Figura de 1×2 subplots comparando 1 capa vs 2 capas,
    con los tres optimizadores en cada panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    for ax, (results, title) in zip(
        axes,
        [(results_1layer, "1 capa oculta"),
         (results_2layer, "2 capas ocultas")]
    ):
        for opt_name, res in results.items():
            losses = res["loss_per_epoch"]
            epochs = range(1, len(losses) + 1)
            ax.plot(
                epochs, losses,
                color=COLORS[opt_name],
                linestyle=LINE_STYLES[opt_name],
                linewidth=2,
                marker="o", markersize=5,
                label=opt_name
            )
        ax.set_title(f"Word2Vec — {title}", fontsize=12)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Pérdida", fontsize=10)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, linestyle=":")

    fig.suptitle("Comparación de optimizadores: SGD vs RMSProp vs Adam", fontsize=13)
    fig.tight_layout()

    if save:
        fname = RESULTS_DIR / "convergencia_comparacion.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {fname}")

    plt.close()
    return fig


# ── 2. Visualización t-SNE de embeddings ─────────────────────

def plot_tsne(embeddings: np.ndarray, vocab, n_words: int = 200,
              title: str = "Embeddings Word2Vec (t-SNE)", save: bool = True):
    """
    Reduce los embeddings a 2D con t-SNE y grafica las palabras
    más frecuentes del vocabulario.

    Parámetros
    ----------
    embeddings : np.ndarray  (vocab_size × embed_dim)
    vocab      : Vocabulary
    n_words    : int         Número de palabras a visualizar
    """
    from sklearn.manifold import TSNE

    n_words = min(n_words, len(vocab.idx2word))
    emb_subset = embeddings[:n_words]
    words      = [vocab.idx2word[i] for i in range(n_words)]

    print(f"Calculando t-SNE para {n_words} palabras...")
    tsne   = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(emb_subset)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.5, color="#185FA5")

    # Etiquetar solo las primeras 60 palabras para no saturar
    for i, word in enumerate(words[:60]):
        ax.annotate(word, coords[i], fontsize=7, alpha=0.8)

    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()

    if save:
        fname = RESULTS_DIR / "tsne_embeddings.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {fname}")

    plt.close()
    return fig


# ── 3. Tabla comparativa de resultados ───────────────────────

def print_results_table(all_results: list[dict]):
    """
    Imprime una tabla comparativa de todas las configuraciones.

    Parámetros
    ----------
    all_results : list de dicts, cada uno con:
        { "arquitectura", "optimizador", "accuracy", "f1_macro",
          "tiempo_total" }
    """
    header = f"{'Arquitectura':<18} {'Optimizador':<12} {'lr':>8} {'Accuracy':>9} {'F1-macro':>9} {'Tiempo(s)':>10}"
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))
    for r in all_results:
        lr_str = f"{r['lr']:g}" if r.get("lr") is not None else "—"
        print(
            f"{r['arquitectura']:<18} "
            f"{r['optimizador']:<12} "
            f"{lr_str:>8} "
            f"{r['accuracy']:>9.4f} "
            f"{r['f1_macro']:>9.4f} "
            f"{r['tiempo_total']:>10.1f}"
        )
    print("═" * len(header) + "\n")