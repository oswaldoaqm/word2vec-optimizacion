#!/usr/bin/env bash
# ============================================================
#  setup.sh - Configura el entorno del proyecto Word2Vec
#  Uso: bash setup.sh
# ============================================================

set -e

echo "[1/5] Creando entorno virtual..."
python -m venv venv

echo "[2/5] Activando entorno..."
# Detectar sistema operativo
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate   # Git Bash en Windows
else
    source venv/bin/activate       # Linux / macOS
fi

echo "[3/5] Actualizando pip..."
pip install --upgrade pip

echo "[4/5] Instalando dependencias base..."
pip install datasets gensim scikit-learn matplotlib seaborn numpy pandas tqdm optuna

echo "[5/5] Instalando PyTorch..."
echo ""
echo "Selecciona tu hardware:"
echo "  1. GPU NVIDIA con CUDA 12.1 (RTX 30xx / 40xx)"
echo "  2. GPU NVIDIA con CUDA 11.8 (RTX 20xx / 30xx)"
echo "  3. Solo CPU (sin GPU)"
echo ""
read -p "Ingresa 1, 2 o 3: " opcion

if [ "$opcion" = "1" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [ "$opcion" = "2" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision
fi

echo ""
echo "Verificando instalacion..."
python -c "
import torch, datasets, gensim, sklearn, matplotlib
print('  torch    :', torch.__version__)
print('  CUDA OK  :', torch.cuda.is_available())
print('  datasets :', datasets.__version__)
print('  gensim   :', gensim.__version__)
print('  sklearn  :', sklearn.__version__)
print('Todo listo.')
"

echo ""
echo "Para activar el entorno en el futuro:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  source venv/Scripts/activate"
else
    echo "  source venv/bin/activate"
fi

echo ""
echo "Nota: exp6.py requiere los embeddings SBWC (descarga manual, ~1GB):"
echo "  mkdir -p embeddings"
echo "  curl -L -o embeddings/SBW-vectors-300-min5.bin.gz \\"
echo "    http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz"