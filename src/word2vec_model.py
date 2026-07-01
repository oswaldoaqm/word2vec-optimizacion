import torch
import torch.nn as nn
import numpy as np
from collections import Counter


# ── Vocabulario ──────────────────────────────────────────────

class Vocabulary:
    """
    Construye el vocabulario a partir del corpus tokenizado.

    Parámetros
    ----------
    min_count : int   Ignora palabras que aparecen menos de min_count veces
    """
    def __init__(self, min_count: int = 5):
        self.min_count = min_count
        self.word2idx:  dict[str, int] = {}
        self.idx2word:  dict[int, str] = {}
        self.vocab_size: int = 0

    def build(self, tokenized_corpus: list[list[str]]):
        counts = Counter(w for sent in tokenized_corpus for w in sent)
        words  = [w for w, c in counts.items() if c >= self.min_count]

        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(words)

        print(f"Vocabulario: {self.vocab_size:,} palabras "
              f"(min_count={self.min_count})")
        return self

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.word2idx[t] for t in tokens if t in self.word2idx]


# ── Arquitectura: 1 capa oculta (Word2Vec clásico) ───────────

class Word2Vec1Layer(nn.Module):
    """
    Arquitectura Skip-gram clásica con una sola capa oculta.

    Parámetros
    ----------
    vocab_size   : int   Tamaño del vocabulario
    embed_dim    : int   Dimensión de los embeddings (hiperparámetro)

    La matriz de embeddings de entrada (W_in) contiene
    las representaciones finales de cada palabra.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 100):
        super().__init__()
        self.embed_dim   = embed_dim
        self.vocab_size  = vocab_size

        # W_in : (vocab_size × embed_dim)  — embeddings de entrada
        # W_out: (vocab_size × embed_dim)  — embeddings de contexto
        self.W_in  = nn.Embedding(vocab_size, embed_dim)
        self.W_out = nn.Embedding(vocab_size, embed_dim)

        # Inicialización uniforme pequeña (estándar en Word2Vec)
        nn.init.uniform_(self.W_in.weight,  -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.W_out.weight)

    def forward(self, center_words, context_words, negative_words):
        """
        Negative Sampling loss.

        Parámetros
        ----------
        center_words   : (B,)     índices de palabras centrales
        context_words  : (B,)     índices de palabras de contexto (positivos)
        negative_words : (B, K)   índices de palabras negativas muestreadas

        Retorna
        -------
        loss : escalar (media del batch)
        """
        # Embeddings de entrada para palabras centrales
        h = self.W_in(center_words)                         # (B, D)

        # Parte positiva: log σ(h · w_context)
        w_pos  = self.W_out(context_words)                  # (B, D)
        score_pos = torch.sum(h * w_pos, dim=1)             # (B,)
        loss_pos  = -torch.log(torch.sigmoid(score_pos) + 1e-9)

        # Parte negativa: sum_k log σ(-h · w_neg_k)
        w_neg  = self.W_out(negative_words)                 # (B, K, D)
        score_neg = torch.bmm(w_neg, h.unsqueeze(2)).squeeze(2)  # (B, K)
        loss_neg  = -torch.sum(
            torch.log(torch.sigmoid(-score_neg) + 1e-9), dim=1   # (B,)
        )

        return (loss_pos + loss_neg).mean()

    def get_embeddings(self) -> np.ndarray:
        """Devuelve la matriz W_in como numpy array (vocab_size × embed_dim)."""
        return self.W_in.weight.detach().cpu().numpy()


# ── Arquitectura: 2 capas ocultas (extensión) ────────────────

class Word2Vec2Layer(nn.Module):
    """
    Word2Vec extendido con dos capas ocultas.

    La primera capa proyecta el embedding inicial a un espacio
    intermedio; la segunda produce la representación final usada
    para calcular la similitud con las palabras de contexto.

    Esto convierte el problema de optimización en no-convexo:
    la pérdida tiene múltiples mínimos locales y es sensible
    al learning rate — lo que justifica comparar optimizadores.

    Parámetros
    ----------
    vocab_size   : int   Tamaño del vocabulario
    embed_dim    : int   Dimensión de los embeddings (capa 1)
    hidden_dim   : int   Dimensión de la capa oculta intermedia (capa 2)
    """
    def __init__(self, vocab_size: int, embed_dim: int = 100, hidden_dim: int = 64):
        super().__init__()
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Capa 1: embedding de entrada
        self.W_in   = nn.Embedding(vocab_size, embed_dim)

        # Capa 2: transformación no-lineal adicional
        self.hidden = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh()               # activación no-lineal → landscape no-convexo
        )

        # Embeddings de contexto (salida)
        self.W_out  = nn.Embedding(vocab_size, hidden_dim)

        nn.init.uniform_(self.W_in.weight,  -0.5 / embed_dim,  0.5 / embed_dim)
        nn.init.zeros_(self.W_out.weight)
        nn.init.xavier_uniform_(self.hidden[0].weight)

    def forward(self, center_words, context_words, negative_words):
        # Embedding + transformación oculta
        emb = self.W_in(center_words)          # (B, embed_dim)
        h   = self.hidden(emb)                 # (B, hidden_dim)

        # Positivos
        w_pos     = self.W_out(context_words)  # (B, hidden_dim)
        score_pos = torch.sum(h * w_pos, dim=1)
        loss_pos  = -torch.log(torch.sigmoid(score_pos) + 1e-9)

        # Negativos
        w_neg     = self.W_out(negative_words)                       # (B, K, D)
        score_neg = torch.bmm(w_neg, h.unsqueeze(2)).squeeze(2)      # (B, K)
        loss_neg  = -torch.sum(
            torch.log(torch.sigmoid(-score_neg) + 1e-9), dim=1
        )

        return (loss_pos + loss_neg).mean()

    def get_embeddings(self) -> np.ndarray:
        """
        El embedding final de cada palabra es el resultado de pasar
        W_in por la capa oculta (la representación aprendida en 2 pasos).
        """
        with torch.no_grad():
            W = self.W_in.weight                    # (V, embed_dim)
            H = self.hidden(W)                      # (V, hidden_dim)
        return H.cpu().numpy()
