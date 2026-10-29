@echo off
chcp 65001 >nul
cd /d "%~dp0ai-doc-platform\backend"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] venv not found at ai-doc-platform\backend\.venv
    echo Please run:  python -m venv .venv
    pause
    exit /b 1
)

.venv\Scripts\python.exe check.py
echo.
pause
