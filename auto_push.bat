@echo off
echo ==============================================
echo         Git Auto Push Script
echo ==============================================

if not exist ".git" (
    echo [ERROR] Not a git repository. Run git init first.
    pause
    exit /b 1
)

echo [1/4] Current git status:
git status -s

echo.
echo [2/4] Adding all changes...
git add .

:: Generate timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set commit_msg=Auto commit %datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2% %datetime:~8,2%:%datetime:~10,2%

echo.
echo [3/4] Committing with message: %commit_msg%
git commit -m "%commit_msg%"

if %errorlevel% neq 0 (
    echo Nothing to commit, skipping push.
) else (
    echo.
    echo [4/4] Pushing to origin/main...
    git push -u origin main
    if %errorlevel% equ 0 (
        echo.
        echo ==============================================
        echo Push successful!
        echo ==============================================
    ) else (
        echo.
        echo ==============================================
        echo Push failed. Check network or remote config.
        echo ==============================================
    )
)

pause