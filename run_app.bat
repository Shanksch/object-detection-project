@echo off
setlocal
title Object Detection Web Application

REM Resolve repo root even when invoked via shortcut
set "SCRIPT_ROOT=%~dp0"
set "VENV_PYTHON=%SCRIPT_ROOT%venv\Scripts\python.exe"

REM Prefer bundled venv; fallback to global python on PATH
if exist "%VENV_PYTHON%" (
    set "PYTHON_EXE=%VENV_PYTHON%"
) else (
    set "PYTHON_EXE=python"
)

echo Using "%PYTHON_EXE%"
echo Starting Flask app...
"%PYTHON_EXE%" "%SCRIPT_ROOT%app_clean.py"

echo.
echo 👋 Application stopped.
pause
