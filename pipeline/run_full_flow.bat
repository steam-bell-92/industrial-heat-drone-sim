@echo off
setlocal

set REPO_ROOT=%~dp0..
set PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe

if "%~1"=="" (
  echo Usage: run_full_flow.bat MAP_IMAGE_PATH PROJECT_ID [CLIENT_NAME]
  exit /b 1
)

if "%~2"=="" (
  echo Usage: run_full_flow.bat MAP_IMAGE_PATH PROJECT_ID [CLIENT_NAME]
  exit /b 1
)

set MAP_IMAGE=%~1
set PROJECT_ID=%~2
set CLIENT_NAME=%~3
if "%CLIENT_NAME%"=="" set CLIENT_NAME=unknown-client

if not defined KAGGLE_API_TOKEN (
  echo KAGGLE_API_TOKEN is not set.
  echo Set it first in this terminal, then re-run.
  exit /b 1
)

"%PYTHON_EXE%" "%REPO_ROOT%\pipeline\full_automation.py" ^
  --repo-root "%REPO_ROOT%" ^
  --map-image "%MAP_IMAGE%" ^
  --project-id "%PROJECT_ID%" ^
  --client-name "%CLIENT_NAME%" ^
  --wait-kernel

endlocal
