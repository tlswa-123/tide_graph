@echo off
title 声音可视化游戏启动器
color 0A

echo ====================================
echo    声音可视化游戏 - 纪念碑谷风格
echo ====================================
echo.

echo 正在检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

echo Python环境检查通过
echo.

echo 选择运行模式：
echo [1] 简化版 (推荐，兼容性好)
echo [2] 完整版 (需要更多依赖)
echo [3] 安装依赖
echo [4] 退出
echo.

set /p choice="请选择 (1-4): "

if "%choice%"=="1" goto simple
if "%choice%"=="2" goto full
if "%choice%"=="3" goto install
if "%choice%"=="4" goto end
echo 无效选择，默认运行简化版
goto simple

:simple
echo.
echo 启动简化版游戏...
echo 正在检查pygame...
python -c "import pygame" >nul 2>&1
if errorlevel 1 (
    echo pygame未安装，正在安装...
    pip install pygame numpy
)
echo 启动游戏...
python simple_game.py
goto end

:full
echo.
echo 启动完整版游戏...
echo 正在检查依赖...
python -c "import pygame, numpy, pyaudio, moderngl" >nul 2>&1
if errorlevel 1 (
    echo 缺少依赖，请先选择选项3安装依赖
    pause
    goto end
)
echo 启动游戏...
python main.py
goto end

:install
echo.
echo 正在安装依赖...
pip install -r requirements.txt
echo 依赖安装完成！
pause
goto end

:end
echo.
echo 感谢使用！
pause