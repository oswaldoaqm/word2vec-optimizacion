import re
import string
import pickle
from pathlib import Path

import numpy as np

RUTA = Path("modelo_demo") / "modelo.pkl"
if not RUTA.exists():
    raise SystemExit(
        f"No encuentro {RUTA}.\n"
        "Primero corre una vez:  python guardar_modelo_demo.py"
    )

M = pickle.load(open(RUTA, "rb"))
WORD2IDX = M["word2idx"]
EMB      = M["embeddings"]
CLF      = M["clf"]
DIM      = M["embed_dim"]
IDX2WORD = {i: w for w, i in WORD2IDX.items()}


# ── Utilidades del modelo ────────────────────────────────────

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation + "0123456789"))
    return [t for t in text.split() if len(t) > 1]


def vector_documento(tokens):
    vecs = [EMB[WORD2IDX[t]] for t in tokens if t in WORD2IDX]
    return np.mean(vecs, axis=0) if vecs else np.zeros(DIM)


def _cos(vec):
    return (EMB @ vec) / (np.linalg.norm(EMB, axis=1) * np.linalg.norm(vec) + 1e-9)


def _falta(*palabras):
    """Devuelve la primera palabra que no esté en el vocabulario, o None."""
    for w in palabras:
        if w not in WORD2IDX:
            return w
    return None


# ── Opción 1: analizar una reseña ────────────────────────────

def analizar_resena(texto):
    tokens    = preprocess(texto)
    conocidas = [t for t in tokens if t in WORD2IDX]
    x         = vector_documento(tokens).reshape(1, -1)
    pred      = int(CLF.predict(x)[0])
    conf      = float(CLF.predict_proba(x)[0][pred]) * 100
    etiqueta  = "POSITIVA  :)" if pred == 1 else "NEGATIVA  :("

    print(f"\n   La reseña es {etiqueta}   (confianza: {conf:.0f}%)")
    print(f"   El modelo reconoció {len(conocidas)} de {len(tokens)} palabras.")

    w = CLF.coef_[0]
    contribs, vistas = [], set()
    for t in tokens:
        if t in WORD2IDX and t not in vistas:
            vistas.add(t)
            contribs.append((t, float(np.dot(w, EMB[WORD2IDX[t]]))))
    pos = sorted([c for c in contribs if c[1] > 0], key=lambda x: -x[1])[:3]
    neg = sorted([c for c in contribs if c[1] < 0], key=lambda x:  x[1])[:3]
    if pos:
        print("   Palabras que empujan a POSITIVO: " + ", ".join(w for w, _ in pos))
    if neg:
        print("   Palabras que empujan a NEGATIVO: " + ", ".join(w for w, _ in neg))


# ── Opción 2: palabras parecidas ─────────────────────────────

def palabras_parecidas(palabra):
    palabra = palabra.lower()
    if _falta(palabra):
        print(f"\n   '{palabra}' no está en el vocabulario del modelo.")
        return
    sims = _cos(EMB[WORD2IDX[palabra]])
    sims[WORD2IDX[palabra]] = -1.0
    print(f"\n   Palabras más parecidas a '{palabra}':")
    for i in np.argsort(sims)[::-1][:8]:
        print(f"      {IDX2WORD[i]:<18} {sims[i]:.2f}")


# ── Opción 3: comparar dos palabras ──────────────────────────

def comparar(a, b):
    a, b = a.lower(), b.lower()
    falta = _falta(a, b)
    if falta:
        print(f"\n   '{falta}' no está en el vocabulario del modelo.")
        return
    va, vb = EMB[WORD2IDX[a]], EMB[WORD2IDX[b]]
    cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
    nivel = ("MUY parecidas" if cos > 0.5
             else "algo parecidas" if cos > 0.25 else "poco parecidas")
    print(f"\n   '{a}'  y  '{b}'   ->   {cos:.2f}   ({nivel})")


# ── Opción 4: analogía ───────────────────────────────────────

def hacer_analogia(a, b, c):
    a, b, c = a.lower(), b.lower(), c.lower()
    falta = _falta(a, b, c)
    if falta:
        print(f"\n   '{falta}' no está en el vocabulario del modelo.")
        return
    vec = EMB[WORD2IDX[a]] - EMB[WORD2IDX[b]] + EMB[WORD2IDX[c]]
    sims = _cos(vec)
    for w in (a, b, c):
        sims[WORD2IDX[w]] = -1.0
    print(f"\n   {a} − {b} + {c}   ≈")
    for i in np.argsort(sims)[::-1][:5]:
        print(f"      {IDX2WORD[i]:<18} {sims[i]:.2f}")


# ── Opción 5: palabra intrusa ────────────────────────────────

def encontrar_intruso(palabras):
    palabras = [p.lower() for p in palabras]
    falta = _falta(*palabras)
    if falta:
        print(f"\n   '{falta}' no está en el vocabulario del modelo.")
        return
    vecs = np.array([EMB[WORD2IDX[p]] for p in palabras])
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sims = norm @ norm.mean(axis=0)
    print(f"\n   La palabra que NO encaja es:   {palabras[int(np.argmin(sims))]}")


# ── Opción 6: ranking del modelo ─────────────────────────────

def ver_ranking(top=12):
    scores = EMB @ CLF.coef_[0]
    orden = np.argsort(scores)
    print("\n   Palabras más POSITIVAS que el modelo aprendió solo:")
    print("      " + ", ".join(IDX2WORD[i] for i in orden[::-1][:top]))
    print("\n   Palabras más NEGATIVAS:")
    print("      " + ", ".join(IDX2WORD[i] for i in orden[:top]))


# ── Menú ─────────────────────────────────────────────────────

MENU = """
============================================================
     DEMO — Modelo Word2Vec  (analisis de sentimiento)
============================================================
  Elige una opcion escribiendo su numero:

    1  ->  Analizar una reseña        (¿positiva o negativa?)
    2  ->  Palabras parecidas         (a una palabra que elijas)
    3  ->  Comparar dos palabras      (que tan parecidas son)
    4  ->  Analogia                   (excelente - bueno + malo ≈ ?)
    5  ->  Palabra intrusa            (cual no encaja en un grupo)
    6  ->  Ranking del modelo         (palabras + positivas / + negativas)
    0  ->  Salir
"""


def pedir(msg):
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        return "0"


def main():
    print(MENU)
    while True:
        op = pedir("  Opcion > ").lower()

        if op in ("0", "salir", "exit", "quit", ""):
            break

        elif op == "1":
            t = pedir("     Escribe la reseña:  ")
            if t:
                analizar_resena(t)

        elif op == "2":
            p = pedir("     Escribe una palabra:  ")
            if p:
                palabras_parecidas(p.split()[0])

        elif op == "3":
            t = pedir("     Escribe DOS palabras (separadas por espacio):  ").split()
            if len(t) >= 2:
                comparar(t[0], t[1])
            else:
                print("     Necesito dos palabras.")

        elif op == "4":
            print("     (calcula: primera - segunda + tercera)")
            t = pedir("     Escribe TRES palabras:  ").split()
            if len(t) >= 3:
                hacer_analogia(t[0], t[1], t[2])
            else:
                print("     Necesito tres palabras.")

        elif op == "5":
            t = pedir("     Escribe 3 o mas palabras:  ").split()
            if len(t) >= 3:
                encontrar_intruso(t)
            else:
                print("     Necesito al menos tres palabras.")

        elif op == "6":
            ver_ranking()

        else:
            print("     Opcion no valida. Escribe un numero del 0 al 6.")

        input("\n     (Enter para volver al menu) ")
        print(MENU)

    print("\n  ¡Gracias! Fin de la demo.")


if __name__ == "__main__":
    main()
