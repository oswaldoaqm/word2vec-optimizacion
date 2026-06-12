import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from src.data import load_amazon_spanish, preprocess_corpus
from src.word2vec_model import Vocabulary, Word2Vec1Layer
from src.trainer import train_word2vec_with_grad

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS    = 7
MAX_TRAIN = 40000
EMBED_DIM = 100
BATCH     = 512
DEVICE    = "cuda"


def plot_grad_tracking(sgd_res, adam_res, epochs):
    ep = range(1, epochs + 1)
    COLORS = {"SGD": "#E24B4A", "Adam": "#185FA5"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for ax, (name, res) in zip(axes[0], [("SGD", sgd_res), ("Adam", adam_res)]):
        ax.plot(ep, res["loss_per_epoch"],
                color=COLORS[name], linewidth=2.5,
                marker="o", markersize=6)
        ax.set_title(f"Perdida -- {name} (lr=0.001)", fontsize=12)
        ax.set_ylabel("Perdida (Neg. Sampling)")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for ax, (name, res) in zip(axes[1], [("SGD", sgd_res), ("Adam", adam_res)]):
        ax.plot(ep, res["grad_per_epoch"],
                color=COLORS[name], linewidth=2.5,
                marker="s", markersize=6, linestyle="--")
        ax.set_title(f"Norma del gradiente -- {name}", fontsize=12)
        ax.set_ylabel("||grad|| promedio")
        ax.set_xlabel("Epoca")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle(
        "Experimento 2: Por que Adam converge y SGD no?\n"
        "Perdida y norma del gradiente por epoch",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()

    fname = RESULTS_DIR / "exp2_grad_tracking.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {fname}")


def main():
    print("Cargando datos...")
    (train_texts, train_labels, _, _,
     test_texts, test_labels) = load_amazon_spanish(max_train=MAX_TRAIN)
    train_tokens = preprocess_corpus(train_texts)
    vocab = Vocabulary(min_count=5).build(train_tokens)

    print("\n--- SGD lr=0.001 ---")
    model_sgd = Word2Vec1Layer(vocab.vocab_size, embed_dim=EMBED_DIM)
    sgd_res   = train_word2vec_with_grad(
        model_sgd, train_tokens, vocab,
        optimizer_name="SGD", learning_rate=0.001,
        epochs=EPOCHS, batch_size=BATCH,
        window_size=5, num_negatives=5, device=DEVICE
    )

    print("\n--- Adam lr=0.001 ---")
    model_adam = Word2Vec1Layer(vocab.vocab_size, embed_dim=EMBED_DIM)
    adam_res   = train_word2vec_with_grad(
        model_adam, train_tokens, vocab,
        optimizer_name="Adam", learning_rate=0.001,
        epochs=EPOCHS, batch_size=BATCH,
        window_size=5, num_negatives=5, device=DEVICE
    )

    print("\nGenerando figura...")
    plot_grad_tracking(sgd_res, adam_res, EPOCHS)

    print("\nResumen:")
    print(f"  SGD  loss={sgd_res['loss_per_epoch'][-1]:.4f}  grad={sgd_res['grad_per_epoch'][-1]:.4f}")
    print(f"  Adam loss={adam_res['loss_per_epoch'][-1]:.4f}  grad={adam_res['grad_per_epoch'][-1]:.4f}")

    with open(RESULTS_DIR / "exp2_resultados.pkl", "wb") as f:
        pickle.dump({"sgd": sgd_res, "adam": adam_res}, f)

    print("\nExperimento 2 completo.")


if __name__ == "__main__":
    main()