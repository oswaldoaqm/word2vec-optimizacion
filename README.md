# Word2Vec para Analisis de Sentimientos en Espanol

Proyecto Final - DS2011 Optimizacion - Grupo 1 - UTEC

Estudio empirico de Word2Vec para clasificacion binaria de sentimientos
en resenas de productos en espanol, desde una perspectiva de optimizacion
matematica. Se comparan arquitecturas (1 y 2 capas ocultas) y optimizadores
(SGD, RMSProp, Adam), y se ejecutan 6 experimentos adicionales incluyendo
busqueda Bayesiana de hiperparametros con Optuna.

---

## Estructura del proyecto

```
word2vec_proyecto/
-
--- main.py              <- Experimento principal (6 configuraciones)
--- exp1.py              <- Sensibilidad de SGD al learning rate
--- exp2.py              <- Tracking de norma del gradiente
--- exp3.py              <- Baseline TF-IDF + ablacion de dimension
--- exp4.py              <- Analisis semantico de embeddings (t-SNE)
--- exp5.py              <- Grid search de hiperparametros
--- exp5b.py             <- Busqueda Bayesiana con Optuna TPE
-
--- src/
-   --- data.py          <- Carga y preprocesamiento del dataset
-   --- word2vec_model.py <- Arquitecturas Word2Vec (1 y 2 capas)
-   --- skipgram_data.py <- Generacion de pares Skip-gram
-   --- trainer.py       <- Loop de entrenamiento (SGD/RMSProp/Adam)
-   --- classifier.py    <- Clasificador logistico sobre embeddings
-   --- visualization.py <- Curvas de convergencia, t-SNE, tablas
-
--- results/             <- Figuras y resultados generados (auto-creado)
--- requirements.txt     <- Dependencias Python (sin PyTorch)
--- setup.sh             <- Setup automatico Git Bash / Linux / macOS
--- setup.bat            <- Setup automatico CMD Windows
```

---

## Instalacion desde cero

### Opcion A - Setup automatico (recomendado)

**Git Bash o Linux/macOS:**
```bash
bash setup.sh
```

**CMD Windows:**
```cmd
setup.bat
```

### Opcion B - Manual

```bash
# 1. Crear y activar entorno virtual
python -m venv venv
source venv/Scripts/activate   # Git Bash Windows
# source venv/bin/activate     # Linux / macOS

# 2. Instalar dependencias base
pip install -r requirements.txt

# 3. Instalar PyTorch segun tu hardware

# Con GPU NVIDIA CUDA 12.1 (RTX 30xx / 40xx):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Con GPU NVIDIA CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Sin GPU (CPU unicamente - mas lento pero funciona):
pip install torch torchvision

# 4. Verificar instalacion
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Ejecutar los experimentos

Activa el entorno antes de cada sesion:
```bash
source venv/Scripts/activate   # Git Bash
```

### Experimento principal
```bash
# Completo (~40 min con GPU)
python main.py

# Solo 1 capa, prueba rapida (~5 min)
python main.py --epochs 2 --max_train 5000 --skip_2layer
```

### Experimentos adicionales (en orden)
```bash
python exp1.py    # Sensibilidad SGD al lr (~50 min)
python exp2.py    # Norma del gradiente   (~25 min)
python exp3.py    # TF-IDF + ablacion d   (~25 min)
python exp4.py    # Analisis semantico    (~15 min)
python exp5.py    # Grid search           (~35 min)

pip install optuna
python exp5b.py   # Busqueda Bayesiana   (~50 min)
```

---

## Parametros de main.py

| Parametro      | Default | Descripcion                        |
|----------------|---------|------------------------------------|
| `--epochs`     | 5       | Epocas de entrenamiento            |
| `--batch_size` | 512     | Tamano de batch                    |
| `--embed_dim`  | 100     | Dimension de embeddings            |
| `--hidden_dim` | 64      | Dimension capa oculta (2 capas)    |
| `--window`     | 5       | Ventana de contexto Skip-gram      |
| `--negatives`  | 5       | Palabras negativas por par         |
| `--lr`         | 0.001   | Learning rate                      |
| `--max_train`  | 40000   | Maximo muestras de entrenamiento   |
| `--device`     | cuda    | `cuda` para GPU, `cpu` para CPU    |
| `--skip_2layer`| False   | Omitir experimento con 2 capas     |

---

## Resultados principales

| Experimento | Hallazgo |
|---|---|
| Principal | RMSProp F1=0.8942, Adam F1=0.8930, SGD F1=0.6157 |
| Exp1 | SGD con lr=0.1 recupera F1=0.8745 - sensible al lr |
| Exp2 | Norma gradiente SGD es 80x menor que Adam |
| Exp3 | TF-IDF gana con F1=0.9225, codo en d=100-150 |
| Exp4 | excelente->inmejorable(0.82), roto->estropeado(0.74) |
| Exp5 | Grid: lr=0.001 d=150 F1-test=0.9010 |
| Exp5b | Optuna: lr=0.00104 d=195 F1-test=0.8997 |

---

## Hardware de referencia

Experimentos ejecutados en:
- CPU: Intel Core i9-14900HX
- GPU: NVIDIA RTX 4070 Laptop (8GB VRAM)
- RAM: 16GB DDR5
- OS: Windows 11 Pro
- Python 3.12 | PyTorch 2.5.1 | CUDA 12.1

Sin GPU los tiempos se multiplican aproximadamente por 5-10x.

---

## Dataset

`amazon_reviews_multi` (espanol) via MTEB - 160k resenas de Amazon
con etiquetas de 1-5 estrellas. Se usa clasificacion binaria:
estrellas 1-2 = negativo, 4-5 = positivo, 3 = descartado.

El dataset se descarga automaticamente al ejecutar cualquier script.

---

## Integrantes

- Danna Nickol Gala Vasquez
- Christopher Renato Perez Torres
- Daniel Davis Villalobos Carlos
- Paul Ricardo Magui-a Quispe
- Oswaldo Alejandro Quispe Monzon

Docente: Victor Antonio Torrez Castillo
Curso: DS2011 Optimizacion - UTEC, 2026
