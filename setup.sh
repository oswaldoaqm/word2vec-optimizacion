#!/usr/bin/env bash
# ============================================================
#  setup.sh  —  Configura el entorno del proyecto Word2Vec
#  Ejecutar con:  bash setup.sh
# ============================================================

set -e

echo "[1/4] Creando entorno virtual..."
python -m venv venv

echo "[2/4] Activando entorno..."
source venv/Scripts/activate   # Git Bash en Windows
# source venv/bin/activate     # Linux / macOS — descomentar si aplica

echo "[3/4] Instalando dependencias..."
pip install --upgrade pip

# PyTorch con soporte CUDA 12.1 (RTX 4070)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Resto de dependencias
pip install datasets gensim scikit-learn matplotlib seaborn numpy pandas tqdm optuna

echo "[4/4] Verificando instalación..."
python -c "
import torch, datasets, gensim, sklearn, matplotlib
print(f'  torch     : {torch.__version__}')
print(f'  CUDA OK   : {torch.cuda.is_available()}')
print(f'  datasets  : {datasets.__version__}')
print(f'  gensim    : {gensim.__version__}')
print(f'  sklearn   : {sklearn.__version__}')
print('Todo listo.')
"
