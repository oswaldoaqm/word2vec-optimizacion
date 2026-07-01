import argparse
import pickle
from pathlib import Path

import torch

from src.data          import load_amazon_spanish, preprocess_corpus
from src.word2vec_model import Vocabulary, Word2Vec1Layer, Word2Vec2Layer
from src.trainer        import train_word2vec
from src.classifier     import build_document_matrix, train_classifier, evaluate
from src.utils          import set_seed
from src.visualization  import (
    plot_convergence_curves,
    plot_convergence_comparison,
    plot_tsne,
    print_results_table,
)


LR_BY_OPTIMIZER = {
    "SGD":     0.1,
    "RMSProp": 0.001,
    "Adam":    0.001,
}


# ── Argumentos de línea de comandos ─────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Word2Vec — Experimento de optimización")
    p.add_argument("--epochs",      type=int,   default=5,     help="Epochs de entrenamiento")
    p.add_argument("--batch_size",  type=int,   default=512,   help="Tamaño de batch")
    p.add_argument("--embed_dim",   type=int,   default=100,   help="Dimensión de embeddings")
    p.add_argument("--hidden_dim",  type=int,   default=64,    help="Dimensión capa oculta (solo 2 capas)")
    p.add_argument("--window",      type=int,   default=5,     help="Ventana de contexto")
    p.add_argument("--negatives",   type=int,   default=5,     help="Palabras negativas por par")
    p.add_argument("--lr",          type=float, default=None,  help="Learning rate global (si se omite, se usa el lr optimo por optimizador)")
    p.add_argument("--max_train",   type=int,   default=40_000,help="Máximo muestras train")
    p.add_argument("--device",      type=str,   default="cuda",help="cuda | cpu")
    p.add_argument("--skip_2layer", action="store_true",       help="Omitir experimento con 2 capas")
    return p.parse_args()


# ── Pipeline principal ───────────────────────────────────────

def main():
    args = parse_args()
    set_seed(42)
    Path("results").mkdir(exist_ok=True)

    # ── 1. Datos ─────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  PASO 1: Carga y preprocesamiento")
    print("═"*55)

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        load_amazon_spanish(max_train=args.max_train)

    print("Preprocesando corpus...")
    train_tokens = preprocess_corpus(train_texts)
    val_tokens   = preprocess_corpus(val_texts)
    test_tokens  = preprocess_corpus(test_texts)

    # ── 2. Vocabulario ───────────────────────────────────────
    print("\n" + "═"*55)
    print("  PASO 2: Construcción del vocabulario")
    print("═"*55)

    vocab = Vocabulary(min_count=5)
    vocab.build(train_tokens)

    # ── 3. Experimento: 1 capa, 3 optimizadores ─────────────
    print("\n" + "═"*55)
    print("  PASO 3: Entrenamiento Word2Vec — 1 capa oculta")
    print("═"*55)

    results_1layer = {}
    optimizers = ["SGD", "RMSProp", "Adam"]

    for opt in optimizers:
        lr_opt = args.lr if args.lr is not None else LR_BY_OPTIMIZER[opt]
        model = Word2Vec1Layer(vocab.vocab_size, embed_dim=args.embed_dim)
        res   = train_word2vec(
            model, train_tokens, vocab,
            optimizer_name=opt,
            learning_rate=lr_opt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            window_size=args.window,
            num_negatives=args.negatives,
            device=args.device,
        )
        res["lr"]         = lr_opt
        res["model"]      = model
        res["embeddings"] = model.get_embeddings()
        results_1layer[opt] = res

    # ── 4. Experimento: 2 capas, 3 optimizadores ────────────
    results_2layer = {}

    if not args.skip_2layer:
        print("\n" + "═"*55)
        print("  PASO 4: Entrenamiento Word2Vec — 2 capas ocultas")
        print("═"*55)

        for opt in optimizers:
            lr_opt = args.lr if args.lr is not None else LR_BY_OPTIMIZER[opt]
            model = Word2Vec2Layer(
                vocab.vocab_size,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim
            )
            res = train_word2vec(
                model, train_tokens, vocab,
                optimizer_name=opt,
                learning_rate=lr_opt,
                epochs=args.epochs,
                batch_size=args.batch_size,
                window_size=args.window,
                num_negatives=args.negatives,
                device=args.device,
            )
            res["lr"]         = lr_opt
            res["model"]      = model
            res["embeddings"] = model.get_embeddings()
            results_2layer[opt] = res

    # ── 5. Clasificación de sentimientos ─────────────────────
    print("\n" + "═"*55)
    print("  PASO 5: Clasificación de sentimientos")
    print("═"*55)

    all_clf_results = []

    def run_classification(results_dict, arch_name):
        for opt, res in results_dict.items():
            emb = res["embeddings"]
            X_train = build_document_matrix(train_tokens, vocab, emb)
            X_test  = build_document_matrix(test_tokens,  vocab, emb)

            clf     = train_classifier(X_train, train_labels)
            metrics = evaluate(clf, X_test, test_labels,
                               split_name=f"{arch_name} + {opt}")

            all_clf_results.append({
                "arquitectura": arch_name,
                "optimizador":  opt,
                "lr":           res.get("lr"),
                "accuracy":     metrics["accuracy"],
                "f1_macro":     metrics["f1_macro"],
                "tiempo_total": res["total_time"],
            })

    run_classification(results_1layer, "1 capa")
    if results_2layer:
        run_classification(results_2layer, "2 capas")

    # ── 6. Visualizaciones ───────────────────────────────────
    print("\n" + "═"*55)
    print("  PASO 6: Generando visualizaciones")
    print("═"*55)

    plot_convergence_curves(results_1layer, "1 capa")
    if results_2layer:
        plot_convergence_curves(results_2layer, "2 capas")
        plot_convergence_comparison(results_1layer, results_2layer)

    # t-SNE sobre el mejor modelo (Adam, 1 capa)
    best_emb = results_1layer["Adam"]["embeddings"]
    plot_tsne(best_emb, vocab, n_words=300,
              title="Embeddings Word2Vec (Adam, 1 capa) — t-SNE")

    # Tabla final
    print_results_table(all_clf_results)

    # ── 7. Guardar resultados ────────────────────────────────
    print("Guardando resultados en results/resultados.pkl ...")
    with open("results/resultados.pkl", "wb") as f:
        # No guardar los modelos completos para ahorrar espacio
        save_data = {
            "1_capa":  {k: {kk: vv for kk, vv in v.items()
                            if kk not in ("model",)} 
                        for k, v in results_1layer.items()},
            "2_capas": {k: {kk: vv for kk, vv in v.items()
                            if kk not in ("model",)} 
                        for k, v in results_2layer.items()} if results_2layer else {},
            "clasificacion": all_clf_results,
        }
        pickle.dump(save_data, f)

    print("\nExperimento completo. Revisa la carpeta results/")

if __name__ == "__main__":
    main()
