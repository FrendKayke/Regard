@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Installation de Regard ===
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), sys.version" 2>nul
if errorlevel 1 (
  echo Python 3.10 a 3.13 est requis ^(3.12 conseille^) : https://www.python.org/downloads/
  echo Cochez "Add python.exe to PATH" pendant l'installation.
  pause & exit /b 1
)
if not exist .venv %PY% -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 ( echo Echec de l'installation. & pause & exit /b 1 )
echo.
echo Installation terminee. Lancez Regard.bat
pause
