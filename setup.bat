@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (python -m venv .venv) else (py -3 -m venv .venv)
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto fail
echo Setup and offline tests passed. Run start.bat to open EvoLLM.
pause
exit /b 0
:fail
echo Setup failed. Read the error above. Python 3.11 or 3.12 is recommended.
pause
exit /b 1
