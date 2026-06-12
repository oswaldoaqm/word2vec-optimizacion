# src/skipgram_data.py
# ============================================================
#  Generación de pares Skip-gram y muestreo negativo
#
#  Dado un corpus tokenizado y un vocabulario, genera batches
#  de (palabra_central, palabra_contexto, palabras_negativas)
#  para entrenar Word2Vec con Negative Sampling.
# ============================================================

import numpy as np
import torch
from torch.utils.data import IterableDataset


class SkipGramDataset(IterableDataset):
    """
    Dataset iterable que genera pares Skip-gram en tiempo real.

    Parámetros
    ----------
    tokenized_corpus : list[list[str]]   Corpus tokenizado
    vocab            : Vocabulary        Vocabulario construido
    window_size      : int               Tamaño de ventana de contexto
    num_negatives    : int               Palabras negativas por par positivo
    """

    def __init__(self, tokenized_corpus, vocab, window_size=5, num_negatives=5):
        self.corpus        = tokenized_corpus
        self.vocab         = vocab
        self.window_size   = window_size
        self.num_negatives = num_negatives

        # Distribución de muestreo negativo: P(w) ∝ freq(w)^(3/4)
        # (exponente 3/4 es el estándar de Mikolov et al. 2013)
        self._build_neg_table()

    def _build_neg_table(self, table_size: int = 1_000_000):
        """
        Construye una tabla de índices para muestreo negativo eficiente.
        Cada índice aparece proporcional a freq^(3/4).
        """
        counts = np.zeros(self.vocab.vocab_size)
        for sentence in self.corpus:
            for w in sentence:
                if w in self.vocab.word2idx:
                    counts[self.vocab.word2idx[w]] += 1

        counts = counts ** 0.75
        counts /= counts.sum()

        self.neg_table = np.random.choice(
            self.vocab.vocab_size,
            size=table_size,
            p=counts
        )
        self._neg_ptr = 0

    def _sample_negatives(self, center_idx: int, context_idx: int):
        """Muestrea K negativos distintos del centro y del contexto."""
        negs = []
        while len(negs) < self.num_negatives:
            idx = self.neg_table[self._neg_ptr % len(self.neg_table)]
            self._neg_ptr += 1
            if idx != center_idx and idx != context_idx:
                negs.append(idx)
        return negs

    def __iter__(self):
        for sentence in self.corpus:
            indices = self.vocab.encode(sentence)
            if len(indices) < 2:
                continue
            for i, center in enumerate(indices):
                # Ventana dinámica (como en la implementación original)
                w = np.random.randint(1, self.window_size + 1)
                start = max(0, i - w)
                end   = min(len(indices), i + w + 1)

                for j in range(start, end):
                    if j == i:
                        continue
                    context  = indices[j]
                    negatives = self._sample_negatives(center, context)
                    yield center, context, negatives


def collate_skipgram(batch):
    """Función collate para DataLoader: convierte lista de tuplas a tensores."""
    centers   = torch.tensor([b[0] for b in batch], dtype=torch.long)
    contexts  = torch.tensor([b[1] for b in batch], dtype=torch.long)
    negatives = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return centers, contexts, negatives
