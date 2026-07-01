import pickle
from pathlib import Path

from src.data import load_amazon_spanish, preprocess_corpus
from src.word2vec_model import Vocabulary, Word2Vec1Layer
from src.trainer import train_word2vec
from src.classifier import build_document_matrix, train_classifier
from src.utils import set_seed

OUT = Path("modelo_demo")
OUT.mkdir(exist_ok=True)

DIM, OPT, LR, EPOCHS, DEVICE = 200, "Adam", 0.001, 7, "cuda"

MAX_TRAIN = 10_000_000


def main():
    set_seed(42)

    print("1/4  Cargando TODO el dataset de Amazon...  (puede tardar un poco)")
    tr_texts, tr_labels, *_ = load_amazon_spanish(max_train=MAX_TRAIN)
    print(f"     -> {len(tr_texts):,} reseñas para entrenar")
    tr_tokens = preprocess_corpus(tr_texts)

    print("2/4  Construyendo el vocabulario...")
    vocab = Vocabulary(min_count=5).build(tr_tokens)

    print(f"3/4  Entrenando Word2Vec ({OPT}, d={DIM})...  (unos minutos)")
    model = Word2Vec1Layer(vocab.vocab_size, embed_dim=DIM)
    train_word2vec(model, tr_tokens, vocab,
                   optimizer_name=OPT, learning_rate=LR,
                   epochs=EPOCHS, batch_size=512,
                   window_size=5, num_negatives=5, device=DEVICE)
    emb = model.get_embeddings()

    print("4/4  Entrenando el clasificador de sentimiento...")
    X = build_document_matrix(tr_tokens, vocab, emb)
    clf = train_classifier(X, tr_labels)

    with open(OUT / "modelo.pkl", "wb") as f:
        pickle.dump({
            "word2idx":  vocab.word2idx,
            "embeddings": emb,
            "clf":       clf,
            "embed_dim": DIM,
        }, f)

    print(f"\n[OK] Modelo guardado en {OUT / 'modelo.pkl'} "
          f"({emb.shape[0]:,} palabras, d={DIM}).")
    print("Ahora la demo es instantánea:   python demo.py")


if __name__ == "__main__":
    main()
