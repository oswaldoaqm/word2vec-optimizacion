import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from src.data import load_amazon_spanish, preprocess_corpus
from src.word2vec_model import Vocabulary, Word2Vec1Layer
from src.trainer import train_word2vec
from src.classifier import build_document_matrix, train_classifier, evaluate
from src.utils import set_seed

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MAX_TRAIN = 40000
EPOCHS    = 5
BATCH     = 512
DEVICE    = "cuda"

# Grid de busqueda.
# El rango de lr se EXTIENDE por debajo de 0.001 (hasta 1e-4) para que el
# optimo no quede pegado al borde inferior de la rejilla: asi se puede
# afirmar que el grid contiene el optimo en su interior y no en la frontera.
# El rango de d incluye 200 por la misma razon (el optimo ronda d=150).
LR_GRID  = [0.01, 0.003, 0.001, 0.0003, 0.0001]
DIM_GRID = [50, 100, 150, 200]


def run_config(lr, d, train_tokens, train_labels,
               val_tokens, val_labels, vocab):
    """
    Entrena Word2Vec con (lr, d) y evalua en VALIDACION.
    Devuelve f1_macro en val (criterio de seleccion).
    """
    model = Word2Vec1Layer(vocab.vocab_size, embed_dim=d)
    res   = train_word2vec(
        model, train_tokens, vocab,
        optimizer_name="Adam", learning_rate=lr,
        epochs=EPOCHS, batch_size=BATCH,
        window_size=5, num_negatives=5, device=DEVICE
    )
    emb   = model.get_embeddings()

    X_train = build_document_matrix(train_tokens, vocab, emb)
    X_val   = build_document_matrix(val_tokens,   vocab, emb)

    clf     = train_classifier(X_train, train_labels)
    y_pred  = clf.predict(X_val)
    f1_val  = f1_score(val_labels, y_pred, average="macro")

    return {
        "lr": lr, "d": d,
        "f1_val":     f1_val,
        "loss_final": res["loss_per_epoch"][-1],
        "loss_curve": res["loss_per_epoch"],
        "clf":        clf,
        "emb":        emb,
    }


def plot_grid_heatmap(grid_results):
    """
    Heatmap de F1-val para cada combinacion (lr, d).
    """
    lrs  = sorted(set(r["lr"] for r in grid_results), reverse=True)
    dims = sorted(set(r["d"]  for r in grid_results))

    matrix = np.zeros((len(lrs), len(dims)))
    for r in grid_results:
        i = lrs.index(r["lr"])
        j = dims.index(r["d"])
        matrix[i, j] = r["f1_val"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="Blues", vmin=matrix.min() - 0.02, vmax=matrix.max() + 0.01)
    plt.colorbar(im, ax=ax, label="F1-macro (validacion)")

    ax.set_xticks(range(len(dims)))
    ax.set_yticks(range(len(lrs)))
    ax.set_xticklabels([f"d={d}" for d in dims], fontsize=11)
    ax.set_yticklabels([f"lr={lr}" for lr in lrs], fontsize=11)
    ax.set_xlabel("Dimension de embeddings", fontsize=11)
    ax.set_ylabel("Learning rate (Adam)", fontsize=11)
    ax.set_title("Grid Search: F1-macro en validacion\n(test set no tocado aun)",
                 fontsize=12)

    for r in grid_results:
        i = lrs.index(r["lr"])
        j = dims.index(r["d"])
        color = "white" if matrix[i, j] > matrix.mean() else "black"
        ax.text(j, i, f"{r['f1_val']:.4f}",
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

    # Marcar el ganador
    best = max(grid_results, key=lambda r: r["f1_val"])
    bi   = lrs.index(best["lr"])
    bj   = dims.index(best["d"])
    ax.add_patch(plt.Rectangle(
        (bj - 0.5, bi - 0.5), 1, 1,
        fill=False, edgecolor="#E24B4A", linewidth=3,
        label=f"Ganador: lr={best['lr']}, d={best['d']}"
    ))
    ax.legend(fontsize=10, loc="upper right",
              bbox_to_anchor=(1.0, -0.12))

    fig.tight_layout()
    fname = RESULTS_DIR / "exp5_grid_heatmap.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {fname}")


def plot_val_vs_test(best, test_f1):
    """
    Barra comparando F1 validacion vs F1 test del ganador.
    Demuestra que no hay sobreajuste al conjunto de validacion.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    vals   = [best["f1_val"], test_f1]
    labels = ["Validacion\n(usado para seleccion)", "Test\n(evaluacion final)"]
    colors = ["#185FA5", "#E24B4A"]
    bars   = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white")
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("F1-macro")
    ax.set_title(
        f"Ganador: lr={best['lr']}, d={best['d']}\n"
        "Validacion vs Test (sin data leakage)",
        fontsize=11
    )
    ax.grid(True, alpha=0.3, linestyle=":", axis="y")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = RESULTS_DIR / "exp5_val_vs_test.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {fname}")


def main():
    set_seed(42)
    print("Cargando datos...")
    (train_texts, train_labels,
     val_texts,   val_labels,
     test_texts,  test_labels) = load_amazon_spanish(max_train=MAX_TRAIN)

    train_tokens = preprocess_corpus(train_texts)
    val_tokens   = preprocess_corpus(val_texts)
    test_tokens  = preprocess_corpus(test_texts)
    vocab = Vocabulary(min_count=5).build(train_tokens)

    print(f"\nGrid search: {len(LR_GRID)} lr x {len(DIM_GRID)} dims = {len(LR_GRID)*len(DIM_GRID)} combinaciones")
    print("Criterio de seleccion: F1-macro en VALIDACION")
    print("Test set: bloqueado hasta seleccion final\n")
    print("=" * 55)

    grid_results = []
    combo_num    = 0

    for lr in LR_GRID:
        for d in DIM_GRID:
            combo_num += 1
            print(f"\n[{combo_num}/{len(LR_GRID)*len(DIM_GRID)}] lr={lr}  d={d}")
            print("-" * 40)
            res = run_config(lr, d, train_tokens, train_labels,
                             val_tokens, val_labels, vocab)
            grid_results.append(res)
            print(f"  >> F1-val = {res['f1_val']:.4f}  loss = {res['loss_final']:.4f}")

    # Tabla de resultados en validacion
    print("\n" + "=" * 55)
    print("RESULTADOS EN VALIDACION (test bloqueado)")
    print("=" * 55)
    print(f"  {'lr':<8} {'d':<6} {'F1-val':>9} {'Loss final':>11}")
    print("  " + "-" * 36)
    for r in sorted(grid_results, key=lambda x: x["f1_val"], reverse=True):
        print(f"  {str(r['lr']):<8} {r['d']:<6} {r['f1_val']:>9.4f} {r['loss_final']:>11.4f}")

    # Seleccion del ganador
    best = max(grid_results, key=lambda r: r["f1_val"])
    print(f"\n  GANADOR: lr={best['lr']}  d={best['d']}  F1-val={best['f1_val']:.4f}")

    # Evaluacion final en TEST (una sola vez)
    print("\n" + "=" * 55)
    print("EVALUACION FINAL EN TEST (una sola vez)")
    print("=" * 55)
    X_test  = build_document_matrix(test_tokens, vocab, best["emb"])
    metrics = evaluate(best["clf"], X_test, test_labels,
                       split_name=f"GANADOR lr={best['lr']} d={best['d']}")
    test_f1 = metrics["f1_macro"]

    print(f"\n  F1-val  (seleccion) : {best['f1_val']:.4f}")
    print(f"  F1-test (final)     : {test_f1:.4f}")
    diff = abs(best["f1_val"] - test_f1)
    print(f"  Diferencia val-test : {diff:.4f}  ", end="")
    if diff < 0.01:
        print("(sin sobreajuste al conjunto de validacion)")
    else:
        print("(diferencia notable -- revisar)")

    # Figuras
    print("\nGenerando figuras...")
    plot_grid_heatmap(grid_results)
    plot_val_vs_test(best, test_f1)

    # Guardar
    save = {
        "grid_results": [{k: v for k, v in r.items()
                          if k not in ("clf", "emb")}
                         for r in grid_results],
        "best_lr":    best["lr"],
        "best_d":     best["d"],
        "best_f1_val": best["f1_val"],
        "test_f1":    test_f1,
    }
    with open(RESULTS_DIR / "exp5_resultados.pkl", "wb") as f:
        pickle.dump(save, f)

    print("\nResumen final:")
    print(f"  Mejor config  : lr={best['lr']}, d={best['d']}")
    print(f"  F1-validacion : {best['f1_val']:.4f}")
    print(f"  F1-test final : {test_f1:.4f}")
    print("\nExperimento 5 completo.")


if __name__ == "__main__":
    main()