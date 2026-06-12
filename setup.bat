@echo off
REM ============================================================
REM  setup.bat  —  Configura el entorno del proyecto Word2Vec
REM  Ejecutar desde Git Bash o CMD en la carpeta del proyecto
REM ============================================================

echo [1/4] Creando entorno virtual...
python -m venv venv

echo [2/4] Activando entorno...
call venv\Scripts\activate.bat

echo [3/4] Instalando dependencias...
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install datasets gensim scikit-learn matplotlib seaborn numpy pandas tqdm

echo [4/4] Listo. Para activar el entorno en el futuro:
echo        call venv\Scripts\activate.bat   (CMD)
echo        source venv/Scripts/activate     (Git Bash)
pause
