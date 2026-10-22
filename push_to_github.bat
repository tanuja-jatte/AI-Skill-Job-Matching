@echo off
title Push Project to GitHub
echo ====================================================================
echo        Pushing AI-Skill-Job-Matching to GitHub
echo ====================================================================
echo.
echo Target Repository: https://github.com/tanuja-jatte/AI-Skill-Job-Matching
echo.
git push -u origin main
echo.
if %errorlevel% equ 0 (
    echo ====================================================================
    echo SUCCESS! Project successfully pushed to GitHub.
    echo View online: https://github.com/tanuja-jatte/AI-Skill-Job-Matching
    echo ====================================================================
) else (
    echo An error occurred. Please check your internet connection or GitHub login.
)
echo.
pause
