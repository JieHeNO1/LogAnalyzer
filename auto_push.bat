@echo off
chcp 65001 >nul
echo ==============================================
echo            Git 自动推送脚本
echo ==============================================

:: 检查是否为 Git 仓库
if not exist ".git" (
    echo [错误] 当前目录不是 Git 仓库，请先执行 git init。
    pause
    exit /b 1
)

:: 显示当前状态
echo [1/4] 当前 Git 状态：
git status -s

:: 添加所有更改
echo.
echo [2/4] 添加所有更改到暂存区...
git add .

:: 生成带时间戳的提交信息
for /f "tokens=1-3 delims=/- " %%a in ('date /t') do (
    set year=%%a
    set month=%%b
    set day=%%c
)
for /f "tokens=1-2 delims=: " %%a in ('time /t') do (
    set hour=%%a
    set minute=%%b
)
set commit_msg=Auto commit %year%-%month%-%day% %hour%:%minute%

echo.
echo [3/4] 提交更改，信息: %commit_msg%
git commit -m "%commit_msg%"

:: 推送（自动处理远程分支）
echo.
echo [4/4] 推送到远程仓库 origin/main ...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ==============================================
    echo 推送成功！按任意键退出。
    echo ==============================================
) else (
    echo.
    echo ==============================================
    echo 推送失败，请检查网络或远程仓库配置。
    echo ==============================================
)
pause