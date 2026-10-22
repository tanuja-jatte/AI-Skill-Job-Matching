@echo off
title AI-Based Skill-to-Job Matching System
echo ====================================================================
echo      AI-Based Skill-to-Job Matching System - 2nd Year B.Tech
echo ====================================================================
echo.

:: Check Python availability
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Checking user-level Python installation...
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        echo Found Python at %PYTHON_EXE%
    ) else (
        echo Please ensure Python 3.10+ is installed.
        pause
        exit /b 1
    )
) else (
    set "PYTHON_EXE=python"
)

echo [1/3] Verifying dependencies from requirements.txt...
"%PYTHON_EXE%" -m pip install -r requirements.txt --quiet

echo.
echo [2/3] Initializing SQLite database and executing Data Engineering pipeline...
"%PYTHON_EXE%" -c "from utils.helper import run_pipeline; print(run_pipeline()['message'])"

echo.
echo [3/3] Launching Streamlit web application...
echo Opening in default browser at http://localhost:8501
echo.
"%PYTHON_EXE%" -m streamlit run app.py

pause
