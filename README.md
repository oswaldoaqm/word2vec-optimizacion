# Word2Vec para Análisis de Sentimientos en Español

Proyecto Final — DS2011 Optimización — Grupo 1 — UTEC

Estudio empírico de Word2Vec para clasificación binaria de sentimientos en reseñas
de productos en español, desde la perspectiva de la **optimización matemática**.
Word2Vec (Skip-gram + Negative Sampling) se implementa **desde cero en PyTorch**;
se comparan arquitecturas (1 y 2 capas ocultas) y optimizadores (SGD, RMSProp, Adam),
se estudia su función de pérdida y se **verifica numéricamente su gradiente analítico**.
El estudio culmina enfrentando el mejor modelo entrenado contra los embeddings
preentrenados **SBWC** (~1.5 mil millones de palabras), dentro y fuera de dominio.

---

## Estructura del proyecto

```
word2vec_proyecto/
├── main.py              <- Experimento principal: 1 y 2 capas x SGD/RMSProp/Adam
├── exp1.py              <- Sensibilidad de SGD al learning rate
├── exp2.py              <- Tracking de la norma del gradiente (por qué Adam converge)
├── exp3.py              <- Baseline TF-IDF (unigrama y bigrama) + ablación de dimensión
├── exp4.py              <- Análisis semántico de embeddings (similitud coseno, t-SNE)
├── exp5.py              <- Grid search de hiperparámetros (lr x d)
├── exp5b.py             <- Búsqueda Bayesiana con Optuna (TPE)
├── exp6.py              <- Duelo: nuestro W2V vs SBWC preentrenado (in/out-of-domain)
├── gradcheck.py         <- Verificación numérica del gradiente de Negative Sampling
│
├── src/
│   ├── data.py           <- Carga y preprocesamiento del dataset (Amazon)
│   ├── word2vec_model.py <- Arquitecturas Word2Vec (1 y 2 capas)
│   ├── skipgram_data.py  <- Generación de pares Skip-gram + muestreo negativo
│   ├── trainer.py        <- Loop de entrenamiento (SGD/RMSProp/Adam)
│   ├── classifier.py     <- Clasificador logístico sobre embeddings promediados
│   ├── visualization.py  <- Curvas de convergencia, t-SNE, tablas
│   └── utils.py          <- set_seed() para reproducibilidad
│
├── results/             <- Figuras y resultados .pkl (auto-creado)
├── embeddings/          <- Embeddings SBWC preentrenados (descarga manual, ver exp6)
├── data/                <- Cache del dataset de tweets (auto-creado por exp6)
├── requirements.txt     <- Dependencias Python (sin PyTorch: se instala por hardware)
├── setup.sh             <- Setup automático Git Bash / Linux / macOS
└── setup.bat            <- Setup automático CMD Windows
```

---

## Pipeline de experimentos (la lógica)

Los experimentos no son independientes: forman una secuencia con propósito.

1. **Entender la optimización** — `exp1` (sensibilidad al learning rate) y `exp2`
   (norma del gradiente: el *mecanismo* de por qué los métodos adaptativos avanzan
   donde SGD se estanca a tasas pequeñas).
2. **Construir el mejor modelo** — `main` (arquitectura + optimizador), `exp3`
   (dimensión), `exp5`/`exp5b` (hiperparámetros por grid y Optuna).
3. **Ponerlo a prueba** — `exp6`: el mejor modelo entrenado in-domain contra un
   gigante preentrenado (SBWC), midiendo dentro y fuera de dominio.
4. **Validar la teoría** — `gradcheck`: confirma que el gradiente derivado a mano
   (Ec. 4 del informe) coincide con autograd y con diferencias finitas.

---

## Instalación desde cero

### Opción A — Setup automático (recomendado)

```bash
bash setup.sh        # Git Bash / Linux / macOS
setup.bat            # CMD Windows
```

### Opción B — Manual

```bash
# 1. Entorno virtual
python -m venv venv
source venv/Scripts/activate     # Git Bash Windows
# source venv/bin/activate        # Linux / macOS

# 2. Dependencias base (datasets, gensim, scikit-learn, matplotlib, numpy, optuna, ...)
pip install -r requirements.txt

# 3. PyTorch según tu hardware (NO está en requirements.txt)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU CUDA 12.1
# pip install torch --index-url https://download.pytorch.org/whl/cu118  # GPU CUDA 11.8
# pip install torch                                                     # Solo CPU

# 4. Verificar
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

`gensim` (incluido en requirements) es necesario para cargar los embeddings SBWC en `exp6`.

---

## Ejecutar los experimentos

```bash
source venv/Scripts/activate     # activar el entorno antes de cada sesión

python main.py        # Experimento principal           (~40 min con GPU)
python exp1.py        # Sensibilidad SGD al lr           (~50 min)
python exp2.py        # Norma del gradiente              (~30 min)
python exp3.py        # TF-IDF + ablación de dimensión   (~25 min)
python exp4.py        # Análisis semántico               (~15 min)
python exp5.py        # Grid search (5 lr x 4 d)         (~60 min)
python exp5b.py       # Búsqueda Bayesiana (Optuna)      (~75 min)
python gradcheck.py   # Verificación del gradiente       (segundos, corre en CPU)
```

Prueba rápida del pipeline (sin esperar 40 min):
```bash
python main.py --epochs 2 --max_train 5000 --skip_2layer
```

### exp6 — duelo contra SBWC (requiere descarga manual)

`exp6` necesita los embeddings SBWC preentrenados. Se descargan una sola vez:

```bash
mkdir -p embeddings
curl -L -o embeddings/SBW-vectors-300-min5.bin.gz \
  http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz

python exp6.py        # Duelo W2V vs SBWC               (~10 min)
```

El dataset de tweets out-of-domain (`cardiffnlp/tweet_sentiment_multilingual`, split
español) se descarga automáticamente la primera vez.

---

## Parámetros de `main.py`

| Parámetro       | Default | Descripción                                            |
|-----------------|---------|--------------------------------------------------------|
| `--epochs`      | 5       | Épocas de entrenamiento                                |
| `--batch_size`  | 512     | Tamaño de batch                                        |
| `--embed_dim`   | 100     | Dimensión de embeddings                                |
| `--hidden_dim`  | 64      | Dimensión de la capa oculta (solo 2 capas)             |
| `--window`      | 5       | Ventana de contexto Skip-gram                          |
| `--negatives`   | 5       | Palabras negativas por par                             |
| `--lr`          | (auto)  | Si se omite, usa el lr óptimo por optimizador          |
| `--max_train`   | 40000   | Máximo de muestras de entrenamiento                    |
| `--device`      | cuda    | `cuda` para GPU, `cpu` para CPU                         |
| `--skip_2layer` | False   | Omitir el experimento con 2 capas                      |

**Learning rate por optimizador** (comparación justa): SGD usa `lr=0.1`,
RMSProp y Adam usan `lr=0.001`. Comparar los tres con el mismo lr sería injusto,
porque 0.001 es el rango de Adam/RMSProp pero deja a SGD estancado (ver `exp1`).

---

## Resultados principales

| Experimento | Hallazgo (números reales) |
|---|---|
| **Principal** | Mejor: 1 capa + RMSProp **F1=0.8940** (Adam 0.8932). SGD con su lr (0.1) ya converge: **F1=0.8607**. 2 capas no mejoran a 1 capa. |
| **Exp1** | SGD es muy sensible al lr: lr=0.1 → F1=0.8742; lr=0.01 → 0.7715; lr=0.001 → 0.6212 (estancado). |
| **Exp2** | A lr=0.001, la norma del gradiente de SGD es ~80x menor que la de Adam en la época 1 (decae a ~7x); por eso SGD no escapa de la meseta inicial. |
| **Exp3** | TF-IDF supera a Word2Vec: unigrama **F1=0.9107**, bigrama **0.9225** vs mejor W2V 0.8990 (d=200). Confirma que es el dominio, no los bigramas. |
| **Exp4** | Vecinos coseno coherentes (15/15): excelente→inmejorable(0.79), roto→doblado(0.76), recomiendo→aconsejo(0.77), envio→envío(0.84). |
| **Exp5** | Grid (5 lr x 4 d): ganador lr=0.001, d=200 → **F1-test=0.8972**. Óptimo en el interior del grid. |
| **Exp5b** | Optuna (30 trials): lr=0.00059, d=166 → **F1-test=0.8970** ≈ grid. La búsqueda Bayesiana no mejora significativamente (H5 refutada). |
| **Exp6** | **In-domain (Amazon): nuestro W2V 0.903 > SBWC 0.883.** Out-of-domain (tweets): SBWC 0.759 > W2V 0.713. Cobertura de vocabulario: W2V 82.7% vs SBWC 96.4%. |
| **gradcheck** | El gradiente analítico de Negative Sampling (Ec. 4) coincide con autograd (~1e-16) y con diferencias finitas: **VERIFICADO**. |

---

## Decisiones de diseño (criterio, no azar)

- **lr por optimizador en `main`:** comparar los 3 al mismo lr favorecería a Adam/RMSProp.
  Cada uno se entrena en su rango estable; la sensibilidad de SGD se estudia aparte (`exp1`).
- **Adam en `exp3`–`exp6`:** RMSProp (0.8940) y Adam (0.8932) están empatados (ruido);
  se fija Adam como adaptativo representativo por ser el estándar de facto.
- **d=100 como base, d=300 en `exp6`:** d=100 es el control fijo para comparar
  optimizadores/arquitecturas sin que la dimensión confunda; `exp3`/`exp5`/`exp5b`
  exploran d (óptimo ~150–200). En `exp6` se usa d=300 para igualar la dimensión de
  SBWC y que el duelo sea justo (el F1 ya está en meseta pasado d≈150).
- **7 épocas en `exp1`/`exp2`, 5 en el resto:** SGD a lr pequeño avanza lento; más
  épocas dan resolución para observar su dinámica de convergencia y de gradiente.
- **Tweets (CardiffNLP) como dataset out-of-domain:** MuchoCine ya no está disponible
  en su fuente original; los tweets son un benchmark reconocido (XLM-T, LREC 2022) y
  ofrecen mayor contraste de dominio frente a las reseñas de Amazon.

---

## Datasets y embeddings

- **Entrenamiento / in-domain:** `mteb/amazon_reviews_multi` (español), 160k reseñas
  totales; se usa un subconjunto balanceado de **40k** para entrenar. Binarización:
  estrellas 1–2 = negativo, 4–5 = positivo, 3 = descartado. Descarga automática.
- **Out-of-domain (exp6):** `cardiffnlp/tweet_sentiment_multilingual` (split español),
  tweets de sentimiento. Descarga automática.
- **Embeddings preentrenados (exp6):** SBWC word2vec (Cardellino), 300d,
  1,000,653 vectores, ~1.5B palabras. Descarga manual (ver sección exp6).

---

## Hardware de referencia

- CPU: Intel Core i9-14900HX · GPU: NVIDIA RTX 4070 Laptop (8GB) · RAM: 16GB DDR5
- OS: Windows 11 Pro · Python 3.12 · PyTorch (CUDA 12.1)

Sin GPU los tiempos se multiplican ~5–10x. El cuello de botella es la generación de
pares Skip-gram en CPU, no el cómputo en GPU.

---

## Integrantes

- Danna Nickol Gala Vásquez
- Christopher Renato Perez Torres
- Daniel Davis Villalobos Carlos
- Paul Ricardo Maguiña Quispe
- Oswaldo Alejandro Quispe Monzón

Docente: Victor Antonio Torrez Castillo · Curso: DS2011 Optimización — UTEC, 2026
