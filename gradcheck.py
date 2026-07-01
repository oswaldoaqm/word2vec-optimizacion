import numpy as np
import torch

from src.utils import set_seed


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def ns_loss_clean(u, v_pos, v_neg):
    """
    Perdida de Negative Sampling 'limpia' (sin el epsilon numerico del
    modelo), para un unico par positivo con K negativos.

    u     : (d,)    embedding de entrada de la palabra central
    v_pos : (d,)    embedding de salida de la palabra de contexto
    v_neg : (K, d)  embeddings de salida de los K negativos
    """
    score_pos = torch.dot(v_pos, u)               # escalar
    score_neg = v_neg @ u                          # (K,)
    loss_pos  = -torch.log(sigmoid(score_pos))
    loss_neg  = -torch.sum(torch.log(sigmoid(-score_neg)))
    return loss_pos + loss_neg


def analytical_grads(u, v_pos, v_neg):
    """Gradientes analiticos de Ec. (4) y sus analogos para v."""
    s_pos = sigmoid(torch.dot(v_pos, u))           # sigma(v_O . u)
    s_neg = sigmoid(v_neg @ u)                     # sigma(v_k . u)  (K,)

    # dL/du = (s_pos - 1) v_pos + sum_k s_neg_k v_neg_k
    grad_u = (s_pos - 1.0) * v_pos + (s_neg.unsqueeze(1) * v_neg).sum(dim=0)

    # dL/dv_pos = (s_pos - 1) u
    grad_v_pos = (s_pos - 1.0) * u

    # dL/dv_neg_k = s_neg_k u
    grad_v_neg = s_neg.unsqueeze(1) * u.unsqueeze(0)   # (K, d)

    return grad_u, grad_v_pos, grad_v_neg


def part_a_autograd_vs_analytical(d=16, K=5, tol=1e-9):
    print("=" * 60)
    print("PARTE A — Gradiente analitico (Ec. 4) vs autograd")
    print("=" * 60)

    # doble precision para una comparacion estricta
    u     = torch.randn(d,    dtype=torch.float64, requires_grad=True)
    v_pos = torch.randn(d,    dtype=torch.float64, requires_grad=True)
    v_neg = torch.randn(K, d, dtype=torch.float64, requires_grad=True)

    loss = ns_loss_clean(u, v_pos, v_neg)
    loss.backward()

    g_u, g_vpos, g_vneg = analytical_grads(u.detach(), v_pos.detach(), v_neg.detach())

    diffs = {
        "dL/du      (entrada)":      (u.grad     - g_u).abs().max().item(),
        "dL/dv_pos  (contexto)":     (v_pos.grad - g_vpos).abs().max().item(),
        "dL/dv_neg  (negativos)":    (v_neg.grad - g_vneg).abs().max().item(),
    }
    print(f"  Perdida L = {loss.item():.6f}\n")
    ok = True
    for name, diff in diffs.items():
        estado = "OK" if diff < tol else "FALLA"
        ok = ok and diff < tol
        print(f"  {name:<26} max|autograd - analitico| = {diff:.2e}   [{estado}]")
    print(f"\n  Resultado Parte A: {'TODOS COINCIDEN' if ok else 'HAY DISCREPANCIAS'}")
    return ok


def part_b_finite_differences(d=16, K=5):
    print("\n" + "=" * 60)
    print("PARTE B — Gradiente numerico (diferencias finitas)")
    print("=" * 60)

    u     = torch.randn(d,    dtype=torch.float64, requires_grad=True)
    v_pos = torch.randn(d,    dtype=torch.float64, requires_grad=True)
    v_neg = torch.randn(K, d, dtype=torch.float64, requires_grad=True)

    # gradcheck compara autograd contra diferencias finitas centradas
    ok = torch.autograd.gradcheck(
        ns_loss_clean, (u, v_pos, v_neg),
        eps=1e-6, atol=1e-6, rtol=1e-4,
    )
    print(f"  torch.autograd.gradcheck = {ok}")
    print(f"  Resultado Parte B: {'GRADIENTE NUMERICO CONSISTENTE' if ok else 'FALLA'}")
    return ok


def part_c_model_forward(d=16, K=5, tol=1e-5):
    """Confirma que el forward REAL del modelo (con su epsilon 1e-9)
    produce gradientes equivalentes al analitico: el epsilon es inofensivo."""
    print("\n" + "=" * 60)
    print("PARTE C — Gradiente del modelo real (src/word2vec_model.py)")
    print("=" * 60)

    from src.word2vec_model import Word2Vec1Layer

    V = 50
    model = Word2Vec1Layer(vocab_size=V, embed_dim=d).double()
    # romper la inicializacion en ceros de W_out para tener scores no triviales
    torch.nn.init.normal_(model.W_out.weight, std=0.5)

    center = torch.tensor([7])
    context = torch.tensor([12])
    negatives = torch.tensor([[3, 19, 25, 31, 44][:K]])

    loss = model(center, context, negatives)
    model.zero_grad()
    loss.backward()

    u     = model.W_in.weight[center[0]].detach()
    v_pos = model.W_out.weight[context[0]].detach()
    v_neg = model.W_out.weight[negatives[0]].detach()
    g_u, _, _ = analytical_grads(u, v_pos, v_neg)

    diff = (model.W_in.weight.grad[center[0]] - g_u).abs().max().item()
    estado = "OK" if diff < tol else "FALLA"
    print(f"  Perdida del modelo = {loss.item():.6f}")
    print(f"  max|grad_modelo(u) - analitico(u)| = {diff:.2e}   [{estado}]")
    print(f"  Resultado Parte C: {'EL MODELO IMPLEMENTA Ec. (4)' if diff < tol else 'FALLA'}")
    return diff < tol


def main():
    set_seed(42)
    print("\nVERIFICACION DEL GRADIENTE DE NEGATIVE SAMPLING (Ec. 4)\n")
    a = part_a_autograd_vs_analytical()
    b = part_b_finite_differences()
    c = part_c_model_forward()

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  A) analitico vs autograd      : {'PASA' if a else 'FALLA'}")
    print(f"  B) autograd vs num. (finitas) : {'PASA' if b else 'FALLA'}")
    print(f"  C) modelo real vs analitico   : {'PASA' if c else 'FALLA'}")
    if a and b and c:
        print("\n  El gradiente analitico de la Ec. (4) queda VERIFICADO.")
    else:
        print("\n  Hay verificaciones que NO pasaron: revisar.")


if __name__ == "__main__":
    main()
