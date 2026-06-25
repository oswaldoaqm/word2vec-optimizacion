# src/utils.py
# ============================================================
#  Utilidades de reproducibilidad.
#
#  set_seed() fija todas las fuentes de aleatoriedad usadas en
#  el proyecto: Python random, numpy, y torch (CPU y CUDA).
#  Llamar al inicio de cada script garantiza que las corridas
#  sean reproducibles entre ejecuciones.
# ============================================================

import os
import random

import numpy as np


def set_seed(seed: int = 42):
    """Fija la semilla global para reproducibilidad."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
