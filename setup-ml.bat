@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run setup.bat first.
 pause
 exit /b 1
)
echo Installing local model dependencies. Downloads require several GB of disk space.
".venv\Scripts\python.exe" -m pip install -r requirements-ml.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" scripts\download_models.py
if errorlevel 1 goto fail
".venv\Scripts\python.exe" scripts\configure_ml.py
if errorlevel 1 goto fail
echo Neural setup complete. Run start.bat and ask a question to verify local inference.
pause
exit /b 0
:fail
echo Neural setup did not complete. Source excerpts remain available; read the error above.
pause
exit /b 1
