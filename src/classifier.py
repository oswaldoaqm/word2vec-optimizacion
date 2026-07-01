import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)


# ── Representación de documentos ────────────────────────────

def document_vector(tokens: list[str], vocab, embeddings: np.ndarray) -> np.ndarray:
    """
    Convierte una lista de tokens en un vector de documento
    promediando los embeddings de las palabras conocidas.

    Si ninguna palabra está en el vocabulario, devuelve un vector de ceros.
    """
    vecs = [
        embeddings[vocab.word2idx[t]]
        for t in tokens
        if t in vocab.word2idx
    ]
    if not vecs:
        return np.zeros(embeddings.shape[1])
    return np.mean(vecs, axis=0)


def build_document_matrix(
    tokenized_texts: list[list[str]],
    vocab,
    embeddings: np.ndarray
) -> np.ndarray:
    """
    Construye la matriz de features (N × embed_dim)
    a partir del corpus tokenizado.
    """
    return np.array([
        document_vector(tokens, vocab, embeddings)
        for tokens in tokenized_texts
    ])


# ── Entrenamiento y evaluación ───────────────────────────────

def train_classifier(X_train, y_train, C=1.0, max_iter=1000):
    """
    Entrena una Regresión Logística sobre los embeddings.

    Parámetros
    ----------
    X_train  : np.ndarray  (N × D)
    y_train  : list[int]
    C        : float       Inverso de regularización L2
    max_iter : int

    Retorna
    -------
    clf : LogisticRegression ajustado
    """
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X, y_true, split_name: str = "Test") -> dict:
    """
    Evalúa el clasificador e imprime métricas.

    Retorna diccionario con accuracy, f1_macro, f1_neg, f1_pos.
    """
    y_pred = clf.predict(X)

    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_neg   = f1_score(y_true, y_pred, pos_label=0)
    f1_pos   = f1_score(y_true, y_pred, pos_label=1)
    cm       = confusion_matrix(y_true, y_pred)

    print(f"\n{'─'*45}")
    print(f"Resultados — {split_name}")
    print(f"{'─'*45}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1-macro   : {f1_macro:.4f}")
    print(f"  F1-negativo: {f1_neg:.4f}")
    print(f"  F1-positivo: {f1_pos:.4f}")
    print(f"\nMatriz de confusión:\n{cm}")
    print(f"\nReporte completo:\n"
          f"{classification_report(y_true, y_pred, target_names=['Negativo','Positivo'])}")

    return {
        "accuracy":  acc,
        "f1_macro":  f1_macro,
        "f1_neg":    f1_neg,
        "f1_pos":    f1_pos,
        "confusion": cm,
    }