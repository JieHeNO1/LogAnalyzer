@echo off
:: 强制将控制台代码页设置为 UTF-8，以正确显示中文
chcp 65001 >nul 2>&1

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
:: 使用 %date% 和 %time% 环境变量（注意：中文 Windows 格式可能不同）
set "commit_msg=Auto commit %date:~0,4%-%date:~5,2%-%date:~8,2% %time:~0,2%:%time:~3,2%"
:: 如果小时为个位数，%time% 可能包含前导空格，简单替换一下
set "commit_msg=%commit_msg: =0%"

echo.
echo [3/4] 提交更改，信息: %commit_msg%
git commit -m "%commit_msg%"

:: 如果上一步没有需要提交的更改，git commit 会返回非零值，但不影响推送
if %errorlevel% neq 0 (
    echo 没有需要提交的更改，跳过推送步骤。
) else (
    echo.
    echo [4/4] 推送到远程仓库 origin/main ...
    git push -u origin main
    if %errorlevel% equ 0 (
        echo.
        echo ==============================================
        echo 推送成功！
        echo ==============================================
    ) else (
        echo.
        echo ==============================================
        echo 推送失败，请检查网络或远程仓库配置。
        echo ==============================================
    )
)

pause