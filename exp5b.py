import pickle
import optuna
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
N_TRIALS  = 30          # mas trials => la fase realmente "bayesiana" del TPE
                        # (tras el arranque aleatorio) tiene mas peso

# Variables globales para datos (cargados una sola vez)
TRAIN_TOKENS = None
TRAIN_LABELS = None
VAL_TOKENS   = None
VAL_LABELS   = None
VOCAB        = None


def objective(trial):
    """
    Funcion objetivo para Optuna.
    Cada trial propone (lr, d) y devuelve F1-val.
    Optuna aprende de los resultados anteriores para
    proponer mejores combinaciones (TPE Bayesiano).
    """
    lr = trial.suggest_float("lr", 1e-4, 0.05, log=True)
    d  = trial.suggest_int("d", 50, 256)

    print(f"\n  Trial {trial.number+1}/{N_TRIALS}  lr={lr:.5f}  d={d}")

    model = Word2Vec1Layer(VOCAB.vocab_size, embed_dim=d)
    res   = train_word2vec(
        model, TRAIN_TOKENS, VOCAB,
        optimizer_name="Adam", learning_rate=lr,
        epochs=EPOCHS, batch_size=BATCH,
        window_size=5, num_negatives=5, device=DEVICE
    )
    emb = model.get_embeddings()

    X_train = build_document_matrix(TRAIN_TOKENS, VOCAB, emb)
    X_val   = build_document_matrix(VAL_TOKENS,   VOCAB, emb)

    clf    = train_classifier(X_train, TRAIN_LABELS)
    y_pred = clf.predict(X_val)
    f1_val = f1_score(VAL_LABELS, y_pred, average="macro")

    # Guardar clf y emb en atributos del trial para reutilizar
    trial.set_user_attr("clf", clf)
    trial.set_user_attr("emb", emb)
    trial.set_user_attr("loss_final", res["loss_per_epoch"][-1])

    print(f"  >> F1-val={f1_val:.4f}  loss={res['loss_per_epoch'][-1]:.4f}")
    return f1_val


def plot_optimization_history(study):
    """
    Dos figuras:
    1. Historia de la busqueda: F1-val por trial con la mejor hasta el momento
    2. Importancia de hiperparametros
    """
    trials  = study.trials
    f1_vals = [t.value for t in trials]
    lrs     = [t.params["lr"] for t in trials]
    dims    = [t.params["d"]  for t in trials]

    best_so_far = []
    current_best = -1
    for v in f1_vals:
        if v > current_best:
            current_best = v
        best_so_far.append(current_best)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: historia de la busqueda
    ax = axes[0]
    ax.scatter(range(1, len(f1_vals)+1), f1_vals,
               c="#185FA5", s=60, alpha=0.7, zorder=3,
               label="F1-val de cada trial")
    ax.plot(range(1, len(best_so_far)+1), best_so_far,
            color="#E24B4A", linewidth=2.5,
            label="Mejor hasta el momento")
    ax.set_xlabel("Trial")
    ax.set_ylabel("F1-macro (validacion)")
    ax.set_title("Historia de la busqueda Bayesiana")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")

    # Panel 2: lr de cada trial (log scale)
    ax2 = axes[1]
    colors2 = ["#E24B4A" if v == max(f1_vals) else "#185FA5" for v in f1_vals]
    ax2.scatter(lrs, f1_vals, c=colors2, s=70, alpha=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Learning rate (escala log)")
    ax2.set_ylabel("F1-macro (validacion)")
    ax2.set_title("F1-val vs Learning rate")
    ax2.grid(True, alpha=0.3, linestyle=":")
    best_idx = int(np.argmax(f1_vals))
    ax2.annotate(
        f"mejor\nlr={lrs[best_idx]:.4f}",
        xy=(lrs[best_idx], f1_vals[best_idx]),
        xytext=(lrs[best_idx] * 2, f1_vals[best_idx] - 0.01),
        fontsize=8, color="#E24B4A",
        arrowprops=dict(arrowstyle="->", color="#E24B4A"),
    )

    # Panel 3: dimension de cada trial
    ax3 = axes[2]
    sc = ax3.scatter(dims, f1_vals,
                     c=f1_vals, cmap="Blues",
                     s=70, alpha=0.9, vmin=min(f1_vals), vmax=max(f1_vals))
    plt.colorbar(sc, ax=ax3, label="F1-val")
    ax3.set_xlabel("Dimension de embeddings (d)")
    ax3.set_ylabel("F1-macro (validacion)")
    ax3.set_title("F1-val vs Dimension")
    ax3.grid(True, alpha=0.3, linestyle=":")

    fig.suptitle(
        f"Experimento 5b: Busqueda Bayesiana (Optuna TPE) -- {N_TRIALS} trials",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fname = RESULTS_DIR / "exp5b_optuna_history.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {fname}")


def main():
    global TRAIN_TOKENS, TRAIN_LABELS, VAL_TOKENS, VAL_LABELS, VOCAB

    set_seed(42)
    print("Cargando datos...")
    (train_texts, train_labels,
     val_texts,   val_labels,
     test_texts,  test_labels) = load_amazon_spanish(max_train=MAX_TRAIN)

    TRAIN_TOKENS = preprocess_corpus(train_texts)
    VAL_TOKENS   = preprocess_corpus(val_texts)
    test_tokens  = preprocess_corpus(test_texts)
    TRAIN_LABELS = train_labels
    VAL_LABELS   = val_labels
    VOCAB = Vocabulary(min_count=5).build(TRAIN_TOKENS)

    # Silenciar logs internos de Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\nIniciando busqueda Bayesiana: {N_TRIALS} trials")
    print("  lr  in [1e-4, 0.05]  (log-uniform, continuo)")
    print("  d   in [50, 256]     (entero)")
    print("  Optimizador fijo: Adam")
    print("  Criterio: F1-macro en VALIDACION")
    print("  TEST BLOQUEADO hasta seleccion final\n")
    print("=" * 55)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8),
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

    best_trial = study.best_trial
    best_lr    = best_trial.params["lr"]
    best_d     = best_trial.params["d"]
    best_f1val = best_trial.value
    best_clf   = best_trial.user_attrs["clf"]
    best_emb   = best_trial.user_attrs["emb"]

    print("\n" + "=" * 55)
    print("RESULTADO DE LA BUSQUEDA BAYESIANA")
    print("=" * 55)
    print(f"  Mejor lr : {best_lr:.6f}")
    print(f"  Mejor d  : {best_d}")
    print(f"  F1-val   : {best_f1val:.4f}")

    # Top 5 trials
    print("\n  Top 5 trials:")
    print(f"  {'Trial':>6} {'lr':>10} {'d':>5} {'F1-val':>9}")
    print("  " + "-" * 34)
    top5 = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
    for t in top5:
        marker = " <-- GANADOR" if t.number == best_trial.number else ""
        print(f"  {t.number+1:>6} {t.params['lr']:>10.5f} "
              f"{t.params['d']:>5} {t.value:>9.4f}{marker}")

    # Evaluacion final en TEST (una sola vez)
    print("\n" + "=" * 55)
    print("EVALUACION FINAL EN TEST (una sola vez)")
    print("=" * 55)
    X_test  = build_document_matrix(test_tokens, VOCAB, best_emb)
    metrics = evaluate(best_clf, X_test, test_labels,
                       split_name=f"OPTUNA GANADOR lr={best_lr:.4f} d={best_d}")
    test_f1 = metrics["f1_macro"]

    print(f"\n  F1-val  (seleccion) : {best_f1val:.4f}")
    print(f"  F1-test (final)     : {test_f1:.4f}")
    diff = abs(best_f1val - test_f1)
    print(f"  Diferencia          : {diff:.4f}", end="  ")
    if diff < 0.01:
        print("(sin sobreajuste)")
    else:
        print("(revisar)")

    # Figuras
    print("\nGenerando figuras...")
    plot_optimization_history(study)

    # Guardar
    with open(RESULTS_DIR / "exp5b_resultados.pkl", "wb") as f:
        pickle.dump({
            "best_lr": best_lr, "best_d": best_d,
            "best_f1val": best_f1val, "test_f1": test_f1,
            "all_trials": [
                {"number": t.number+1,
                 "lr": t.params["lr"],
                 "d":  t.params["d"],
                 "f1_val": t.value}
                for t in study.trials
            ],
        }, f)

    print(f"\nResumen final:")
    print(f"  lr optimo  : {best_lr:.6f}")
    print(f"  d optimo   : {best_d}")
    print(f"  F1-val     : {best_f1val:.4f}")
    print(f"  F1-test    : {test_f1:.4f}")
    print("\nExperimento 5b completo.")


if __name__ == "__main__":
    main()