@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv ( echo Lancez d'abord installer.bat & pause & exit /b 1 )
echo.
echo  1. Suivre mon regard
echo  2. Refaire la calibration puis suivre
echo  3. Voir le rapport (dernieres 24 h)
echo  4. Voir le rapport (7 derniers jours)
echo  5. Apercu webcam / gestes du visage
echo.
set /p CHOIX="Votre choix : "
if "%CHOIX%"=="1" .venv\Scripts\python regard.py track
if "%CHOIX%"=="2" .venv\Scripts\python regard.py track --recalibrate
if "%CHOIX%"=="3" .venv\Scripts\python regard.py report
if "%CHOIX%"=="4" .venv\Scripts\python regard.py report --days 7
if "%CHOIX%"=="5" .venv\Scripts\python regard.py preview
pause
