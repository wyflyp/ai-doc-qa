@echo off
chcp 65001 >nul
cd /d "%~dp0ai-doc-platform\backend"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] venv not found at ai-doc-platform\backend\.venv
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

if not exist ".env" (
    echo [ERROR] .env not found. Copy .env.example to .env and fill in DB_PASSWORD / LLM_API_KEY
    pause
    exit /b 1
)

echo ================================================
echo  Backend starting...
echo  API docs : http://127.0.0.1:8000/docs
echo  Health   : http://127.0.0.1:8000/api/health
echo  Keep this window OPEN. Press Ctrl+C to stop.
echo ================================================
echo.

.venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

echo.
echo Backend stopped.
pause
