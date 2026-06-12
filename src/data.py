# src/data.py
import re
import string
from datasets import load_dataset
from torch.utils.data import Dataset


def load_amazon_spanish(max_train=40_000, max_val=5_000, max_test=5_000, seed=42):
    """
    Carga amazon_reviews_multi (mteb, español) y convierte a binario:
      label 0-1  →  negativo (0)
      label 3-4  →  positivo (1)
      label 2    →  descartado (neutro)
    """
    print("Descargando dataset amazon_reviews_multi (español)...")
    ds = load_dataset("mteb/amazon_reviews_multi", "es")

    # Shuffle para mezclar clases (el dataset viene ordenado por label)
    ds = ds.shuffle(seed=seed)

    # Detectar campo de texto
    sample = ds["train"][0]
    text_field = "text" if "text" in sample else "review_body"
    print(f"  Campo texto: '{text_field}' | Labels: 0-4 (0-1=neg, 3-4=pos, 2=neutro)")

    def to_binary(split, max_samples):
        texts, labels = [], []
        for item in ds[split]:
            lbl = item["label"]   # 0,1,2,3,4
            if lbl in (0, 1):
                labels.append(0)      # negativo
            elif lbl in (3, 4):
                labels.append(1)      # positivo
            else:
                continue              # descarta neutro (2)
            texts.append(item[text_field])
            if len(texts) >= max_samples:
                break
        return texts, labels

    train_texts, train_labels = to_binary("train",      max_train)
    val_texts,   val_labels   = to_binary("validation", max_val)
    test_texts,  test_labels  = to_binary("test",       max_test)

    # Verificar balance
    neg_tr = train_labels.count(0)
    pos_tr = train_labels.count(1)
    print(f"  Train : {len(train_texts):,} muestras  (neg={neg_tr:,} pos={pos_tr:,})")
    print(f"  Val   : {len(val_texts):,}  muestras")
    print(f"  Test  : {len(test_texts):,}  muestras")
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+|@\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation + "0123456789"))
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def preprocess_corpus(texts: list[str]) -> list[list[str]]:
    return [preprocess(t) for t in texts]


class SentimentDataset(Dataset):
    def __init__(self, embeddings, labels):
        import torch
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels,     dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]