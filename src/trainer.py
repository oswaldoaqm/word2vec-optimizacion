import time
import torch
from torch.utils.data import DataLoader

from src.skipgram_data import SkipGramDataset, collate_skipgram

OPTIMIZERS = {
    "SGD":      lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
    "RMSProp":  lambda params, lr: torch.optim.RMSprop(params, lr=lr, alpha=0.99),
    "Adam":     lambda params, lr: torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999)),
}


def train_word2vec(
    model,
    tokenized_corpus,
    vocab,
    optimizer_name: str = "Adam",
    learning_rate:  float = 0.001,
    epochs:         int   = 5,
    batch_size:     int   = 512,
    window_size:    int   = 5,
    num_negatives:  int   = 5,
    device:         str   = "cuda",
) -> dict:
    assert optimizer_name in OPTIMIZERS, \
        f"Optimizador desconocido: {optimizer_name}. Opciones: {list(OPTIMIZERS)}"

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    optimizer = OPTIMIZERS[optimizer_name](model.parameters(), learning_rate)

    dataset = SkipGramDataset(
        tokenized_corpus, vocab,
        window_size=window_size,
        num_negatives=num_negatives
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_skipgram,
        num_workers=0
    )

    loss_per_epoch = []
    time_per_epoch = []
    t0_total = time.time()

    print(f"\nEntrenando con {optimizer_name} | lr={learning_rate} | "
          f"device={device} | capas={'2' if hasattr(model, 'hidden') else '1'}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        t0_epoch   = time.time()

        for center, context, negatives in loader:
            center    = center.to(device)
            context   = context.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        epoch_loss = total_loss / max(n_batches, 1)
        epoch_time = time.time() - t0_epoch

        loss_per_epoch.append(epoch_loss)
        time_per_epoch.append(epoch_time)

        print(f"  Epoch {epoch}/{epochs}  |  "
              f"loss: {epoch_loss:.4f}  |  "
              f"tiempo: {epoch_time:.1f}s")

    total_time = time.time() - t0_total
    print(f"-" * 55)
    print(f"Total: {total_time:.1f}s\n")

    return {
        "loss_per_epoch": loss_per_epoch,
        "time_per_epoch": time_per_epoch,
        "total_time":     total_time,
    }


def train_word2vec_with_grad(
    model,
    tokenized_corpus,
    vocab,
    optimizer_name: str = "Adam",
    learning_rate:  float = 0.001,
    epochs:         int   = 5,
    batch_size:     int   = 512,
    window_size:    int   = 5,
    num_negatives:  int   = 5,
    device:         str   = "cuda",
) -> dict:
    assert optimizer_name in OPTIMIZERS

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    optimizer = OPTIMIZERS[optimizer_name](model.parameters(), learning_rate)

    dataset = SkipGramDataset(
        tokenized_corpus, vocab,
        window_size=window_size,
        num_negatives=num_negatives
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_skipgram,
        num_workers=0
    )

    loss_per_epoch = []
    grad_per_epoch = []
    time_per_epoch = []
    t0_total = time.time()

    print(f"\nEntrenando con grad tracking: {optimizer_name} | lr={learning_rate} | device={device}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_grad = 0.0
        n_batches  = 0
        t0_epoch   = time.time()

        for center, context, negatives in loader:
            center    = center.to(device)
            context   = context.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()

            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            optimizer.step()

            total_loss += loss.item()
            total_grad += grad_norm
            n_batches  += 1

        epoch_loss = total_loss / max(n_batches, 1)
        epoch_grad = total_grad / max(n_batches, 1)
        epoch_time = time.time() - t0_epoch

        loss_per_epoch.append(epoch_loss)
        grad_per_epoch.append(epoch_grad)
        time_per_epoch.append(epoch_time)

        print(f"  Epoch {epoch}/{epochs}  |  loss: {epoch_loss:.4f}  |  grad: {epoch_grad:.4f}  |  {epoch_time:.1f}s")

    total_time = time.time() - t0_total
    print(f"  Total: {total_time:.1f}s")

    return {
        "loss_per_epoch": loss_per_epoch,
        "grad_per_epoch": grad_per_epoch,
        "time_per_epoch": time_per_epoch,
        "total_time":     total_time,
    }