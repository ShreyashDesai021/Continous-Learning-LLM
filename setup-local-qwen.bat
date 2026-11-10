@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Existing project environment not found. Run the original setup.bat first.
  pause
  exit /b 1
)
set "EVOLLM_OLLAMA_EXE="
for /f "delims=" %%I in ('where ollama.exe 2^>nul') do if not defined EVOLLM_OLLAMA_EXE set "EVOLLM_OLLAMA_EXE=%%I"
if not defined EVOLLM_OLLAMA_EXE if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "EVOLLM_OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined EVOLLM_OLLAMA_EXE (
  echo Install Ollama for Windows from https://ollama.com/download/windows
  echo Then open Ollama and run this file again.
  pause
  exit /b 1
)
echo Stop the EvoLLM server before configuring. Keep Ollama running.
echo Downloading and importing quantized Qwen2.5-3B, approximately 2.1 GB.
".venv\Scripts\python.exe" scripts\prepare_local_qwen.py --ollama-exe "%EVOLLM_OLLAMA_EXE%"
if errorlevel 1 (
  echo Download failed. Check your internet connection and that Ollama is open.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\configure_local_qwen.py
if errorlevel 1 (
  pause
  exit /b 1
)
echo Start EvoLLM with start.bat and select Local Qwen.
pause
