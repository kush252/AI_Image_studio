@echo off
REM Image Processing Platform -- launcher for cmd.exe.
REM Usage:  run            start everything (API serves the built UI)
REM         run dev        API + Vite dev server, hot reload
REM         run setup      create the venv and install everything
REM         run build      rebuild the UI bundle
REM         run test       backend test suite
setlocal
set ROOT=%~dp0
set PY=%ROOT%.venv\Scripts\python.exe
set TASK=%1
if "%TASK%"=="" set TASK=start

if /I "%TASK%"=="setup" goto setup
if /I "%TASK%"=="build" goto build
if /I "%TASK%"=="test"  goto test
if /I "%TASK%"=="dev"   goto dev
if /I "%TASK%"=="start" goto start
echo Unknown task "%TASK%". Use: setup, build, start, dev, test
exit /b 1

:setup
python -m venv "%ROOT%.venv" || exit /b 1
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r "%ROOT%backend\requirements.txt" || exit /b 1
pushd "%ROOT%frontend" && npm install && popd
goto build

:build
pushd "%ROOT%frontend" && npm run build && popd
exit /b %ERRORLEVEL%

:test
pushd "%ROOT%backend" && "%PY%" -m pytest tests & popd
exit /b %ERRORLEVEL%

:dev
start "IPCV API" cmd /k "cd /d "%ROOT%backend" && "%PY%" -m uvicorn app.main:app --reload --port 8000"
echo API starting on http://127.0.0.1:8000 in a separate window.
echo Use http://localhost:5173 for the hot-reloading UI.
pushd "%ROOT%frontend" && npm run dev & popd
exit /b %ERRORLEVEL%

:start
if not exist "%PY%" echo No virtual environment. Run "run setup" first. & exit /b 1
if not exist "%ROOT%backend\app\static\index.html" call :build
echo.
echo   Image Processing Platform
echo   http://127.0.0.1:8000        the app
echo   http://127.0.0.1:8000/docs   API reference
echo   Ctrl+C to stop
echo.
pushd "%ROOT%backend" && "%PY%" -m uvicorn app.main:app --port 8000 & popd
exit /b %ERRORLEVEL%
