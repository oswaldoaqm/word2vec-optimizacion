@echo off
REM ============================================================
REM  setup.bat - Configura el entorno del proyecto Word2Vec
REM  Ejecutar desde CMD en la carpeta del proyecto
REM ============================================================

echo [1/5] Creando entorno virtual...
python -m venv venv

echo [2/5] Activando entorno...
call venv\Scripts\activate.bat

echo [3/5] Actualizando pip...
pip install --upgrade pip

echo [4/5] Instalando dependencias base...
pip install datasets gensim scikit-learn matplotlib seaborn numpy pandas tqdm optuna

echo [5/5] Instalando PyTorch...
echo.
echo Selecciona tu hardware:
echo   1. GPU NVIDIA con CUDA 12.1 (RTX 30xx / 40xx)
echo   2. GPU NVIDIA con CUDA 11.8 (RTX 20xx / 30xx)
echo   3. Solo CPU (sin GPU)
echo.
set /p opcion="Ingresa 1, 2 o 3: "

if "%opcion%"=="1" (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else if "%opcion%"=="2" (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    pip install torch torchvision
)

echo.
echo Verificando instalacion...
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo.
echo Listo. Para activar el entorno en el futuro:
echo   call venv\Scripts\activate.bat   (CMD)
echo   source venv/Scripts/activate     (Git Bash)
pause