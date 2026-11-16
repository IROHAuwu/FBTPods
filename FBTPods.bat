@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
rem The package lists are copied into .venv only after every package is installed. Until they are
rem there and match packaging\, the install runs (again): a first install that was cut short
rem (window closed, network error) is finished instead of starting a .venv that cannot run the app,
rem and a git pull that changes the lists installs the new versions.
fc /b packaging\requirements.txt .venv\installed-requirements.txt >nul 2>&1 && fc /b packaging\constraints.txt .venv\installed-constraints.txt >nul 2>&1 && goto run
if exist .venv\Scripts\python.exe goto update

echo [FBT Pods] 第一次執行：準備 Python 環境...
set "PY="
rem 64-bit only: onnxruntime and other packages have no 32-bit Windows version
py -3.12-64 -c "import sys" >nul 2>&1 && set "PY=py -3.12-64"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY if exist "%ProgramFiles%\Python312\python.exe" set PY="%ProgramFiles%\Python312\python.exe"
if defined PY goto venv

echo [FBT Pods] 找不到 Python 3.12，正在用 winget 自動安裝（約 1-3 分鐘）...
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY (
  echo.
  echo 自動安裝失敗。請手動安裝 Python 3.12（64 位元）：
  echo   https://www.python.org/downloads/release/python-31210/
  echo 安裝時勾選「Add python.exe to PATH」，裝好後再雙擊 FBTPods.bat。
  start "" https://www.python.org/downloads/release/python-31210/
  pause
  exit /b 1
)

:venv
echo [FBT Pods] 建立環境並安裝套件（第一次約 2-5 分鐘）...
%PY% -m venv .venv
if errorlevel 1 (echo 建立環境失敗 & rmdir /s /q .venv 2>nul & pause & exit /b 1)
goto install

:update
echo [FBT Pods] 安裝或更新套件...
:install
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r packaging\requirements.txt -c packaging\constraints.txt || goto installfail
.venv\Scripts\python.exe -m pip install --no-deps capablerobot-usbregister==1.3.1.2 || goto installfail
copy /y packaging\requirements.txt .venv\installed-requirements.txt >nul
copy /y packaging\constraints.txt .venv\installed-constraints.txt >nul
goto run

:installfail
echo 套件安裝失敗，請把上面的錯誤訊息傳給開發者
rem A .venv that can already run the app (made by an older FBTPods.bat, or before this update) is
rem kept and started: deleting it left nothing that runs until a full reinstall succeeded (offline,
rem or a package file locked by a running FBT Pods). Its package lists stay out of date, so the next
rem start tries the update again. One that cannot run the app is started over.
.venv\Scripts\python.exe -c "import numpy, bumble, vqf" >nul 2>&1 && goto run
rmdir /s /q .venv
pause
exit /b 1

:run
if "%~1"=="" (
  start "" .venv\Scripts\pythonw.exe -m airpodsvr gui
  exit /b 0
)
if "%~1"=="gui" (
  start "" .venv\Scripts\pythonw.exe -m airpodsvr %*
  exit /b 0
)
.venv\Scripts\python.exe -m airpodsvr %*
if errorlevel 1 pause
