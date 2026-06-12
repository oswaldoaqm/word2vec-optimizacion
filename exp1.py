import argparse
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from src.data import load_amazon_spanish, preprocess_corpus
from src.word2vec_model import Vocabulary, Word2Vec1Layer
from src.trainer import train_word2vec
from src.classifier import build_document_matrix, train_classifier, evaluate

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LR_COLORS = {0.1: "#E24B4A", 0.01: "#F5A623", 0.001: "#9B59B6", 0.0001: "#95A5A6"}
LR_LABELS = {0.1: "SGD lr=0.1", 0.01: "SGD lr=0.01", 0.001: "SGD lr=0.001 (original)", 0.0001: "SGD lr=0.0001"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=7)
    parser.add_argument("--max_train", type=int,   default=40000)
    parser.add_argument("--embed_dim", type=int,   default=100)
    parser.add_argument("--batch",     type=int,   default=512)
    parser.add_argument("--device",    type=str,   default="cuda")
    args = parser.parse_args()

    print("Cargando datos...")
    train_texts, train_labels, _, _, test_texts, test_labels = load_amazon_spanish(max_train=args.max_train)
    train_tokens = preprocess_corpus(train_texts)
    test_tokens  = preprocess_corpus(test_texts)
    vocab = Vocabulary(min_count=5).build(train_tokens)

    sgd_results = {}
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        print(f"\n--- SGD lr={lr} ---")
        model = Word2Vec1Layer(vocab.vocab_size, embed_dim=args.embed_dim)
        res = train_word2vec(model, train_tokens, vocab,
                             optimizer_name="SGD", learning_rate=lr,
                             epochs=args.epochs, batch_size=args.batch,
                             window_size=5, num_negatives=5, device=args.device)
        res["embeddings"] = model.get_embeddings()
        sgd_results[lr] = res

    print("\n--- Adam lr=0.001 (referencia) ---")
    model_adam = Word2Vec1Layer(vocab.vocab_size, embed_dim=args.embed_dim)
    adam_ref = train_word2vec(model_adam, train_tokens, vocab,
                              optimizer_name="Adam", learning_rate=0.001,
                              epochs=args.epochs, batch_size=args.batch,
                              window_size=5, num_negatives=5, device=args.device)
    adam_ref["embeddings"] = model_adam.get_embeddings()

    print("\nClasificacion...")
    clf_results = []
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        emb = sgd_results[lr]["embeddings"]
        X_tr = build_document_matrix(train_tokens, vocab, emb)
        X_te = build_document_matrix(test_tokens,  vocab, emb)
        clf = train_classifier(X_tr, train_labels)
        m = evaluate(clf, X_te, test_labels, f"SGD lr={lr}")
        clf_results.append({"lr": lr, "f1": m["f1_macro"],
                             "loss": sgd_results[lr]["loss_per_epoch"][-1]})

    print("\nGenerando figura...")
    ep = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        ax.plot(ep, sgd_results[lr]["loss_per_epoch"],
                color=LR_COLORS[lr], linewidth=2, marker="o", markersize=5,
                label=LR_LABELS[lr])
    ax.set_title("Sensibilidad de SGD al learning rate")
    ax.set_xlabel("Epoca"); ax.set_ylabel("Perdida")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, linestyle=":")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    best_lr = min(sgd_results, key=lambda lr: sgd_results[lr]["loss_per_epoch"][-1])
    ax2 = axes[1]
    ax2.plot(ep, sgd_results[best_lr]["loss_per_epoch"],
             color=LR_COLORS[best_lr], linewidth=2.5, marker="o", markersize=5,
             label=f"SGD lr={best_lr} (mejor SGD)")
    ax2.plot(ep, adam_ref["loss_per_epoch"],
             color="#185FA5", linewidth=2.5, marker="s", markersize=5, linestyle="--",
             label="Adam lr=0.001")
    ax2.set_title("Mejor SGD vs Adam")
    ax2.set_xlabel("Epoca"); ax2.set_ylabel("Perdida")
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle("Experimento 1: Sensibilidad al learning rate", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_lr_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Figura guardada: results/exp1_lr_sensitivity.png")

    print("\nResultados:")
    print(f"  {'lr':<8} {'F1-macro':>9} {'Loss final':>11}")
    print("  " + "-"*30)
    for r in clf_results:
        print(f"  {str(r['lr']):<8} {r['f1']:>9.4f} {r['loss']:>11.4f}")
    print(f"\n  Mejor lr SGD: {best_lr}")
    print("Experimento 1 completo.")

if __name__ == "__main__":
    main()