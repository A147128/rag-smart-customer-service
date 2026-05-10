@echo off
chcp 65001 >nul
echo ========================================
echo   GitHub推送脚本
echo ========================================
echo.

REM 提示用户输入GitHub用户名
set /p GITHUB_USERNAME="请输入你的GitHub用户名: "

echo.
echo 正在添加远程仓库...
git remote add origin https://github.com/%GITHUB_USERNAME%/rag-smart-customer-service.git

if %errorlevel% neq 0 (
    echo.
    echo [错误] 添加远程仓库失败,可能已存在
    echo 尝试更新远程仓库URL...
    git remote set-url origin https://github.com/%GITHUB_USERNAME%/rag-smart-customer-service.git
)

echo.
echo 正在切换分支到main...
git branch -M main

echo.
echo 正在推送到GitHub...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   ✅ 推送成功!
    echo ========================================
    echo.
    echo 下一步:
    echo 1. 访问 https://streamlit.io/cloud
    echo 2. 用GitHub登录
    echo 3. 选择你的仓库进行部署
    echo.
    echo 详细步骤请查看 DEPLOY_GUIDE.md
    echo.
) else (
    echo.
    echo ========================================
    echo   ❌ 推送失败
    echo ========================================
    echo.
    echo 可能的原因:
    echo 1. 仓库名称不正确
    echo 2. GitHub认证失败
    echo 3. 网络连接问题
    echo.
    echo 请检查错误信息并重试
    echo.
)

pause
