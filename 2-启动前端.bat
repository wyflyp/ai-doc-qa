@echo off
chcp 65001 >nul
cd /d "%~dp0ai-doc-frontend"

if not exist "node_modules" (
    echo Installing frontend dependencies, please wait...
    call npm install --registry=https://registry.npmmirror.com
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
)

echo ================================================
echo  Frontend starting...
echo  Open in browser : http://127.0.0.1:5173
echo  Keep this window OPEN. Press Ctrl+C to stop.
echo ================================================
echo.

call npm run dev

echo.
echo Frontend stopped.
pause
