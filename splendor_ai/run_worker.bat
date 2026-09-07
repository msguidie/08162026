@echo off
rem ============================================================
rem  Splendor AI deployment worker — Windows launcher.
rem
rem  Activates the virtualenv, points the worker at splendor_ai\.env and
rem  restarts it whenever it stops for any reason other than "you asked it
rem  to" (exit 0) or "the configuration is wrong" (exit 2).  A sleeping
rem  Render service, a dropped Wi-Fi link or a reboot of the server all end
rem  up here and are simply retried.
rem
rem  Usage:  run_worker.bat            (normal)
rem          run_worker.bat --once     (offline self-test, no restart loop)
rem ============================================================
setlocal enabledelayedexpansion

set "PROJECT=%~dp0"
if "%PROJECT:~-1%"=="\" set "PROJECT=%PROJECT:~0,-1%"
pushd "%PROJECT%\.." >nul
set "REPO=%CD%"
popd >nul

rem -- find the virtualenv: repo root first, then splendor_ai\ --------------
set "PY="
if exist "%REPO%\.venv\Scripts\python.exe"    set "PY=%REPO%\.venv\Scripts\python.exe"
if exist "%PROJECT%\.venv\Scripts\python.exe" set "PY=%PROJECT%\.venv\Scripts\python.exe"
if not defined PY (
  echo [run_worker] No .venv found.  Create one with:
  echo     py -3.11 -m venv .venv
  echo     .venv\Scripts\pip install -r splendor_ai\requirements-worker.txt
  exit /b 2
)

if not exist "%PROJECT%\.env" (
  echo [run_worker] %PROJECT%\.env is missing — copy .env.example to .env first.
  exit /b 2
)

rem The package lives at <repo>\splendor_ai\splendor_ai, imported as
rem `splendor_ai` from the repository root.
set "PYTHONPATH=%REPO%"
set "SPLENDOR_WORKER_ENV=%PROJECT%\.env"
cd /d "%REPO%"

echo [run_worker] python: %PY%
echo [run_worker] config: %SPLENDOR_WORKER_ENV%

rem One-shot modes exit on their own; do not wrap them in the restart loop.
echo %* | findstr /i /c:"--once" /c:"--print-config" /c:"--help" >nul
if not errorlevel 1 (
  "%PY%" -m splendor_ai.worker.worker %*
  exit /b !errorlevel!
)

:loop
"%PY%" -m splendor_ai.worker.worker %*
set "CODE=!errorlevel!"
if "!CODE!"=="0" (
  echo [run_worker] worker stopped cleanly.
  goto :done
)
if "!CODE!"=="2" (
  echo [run_worker] configuration error — fix .env and start again.
  goto :done
)
echo [run_worker] worker exited with code !CODE! — restarting in 5 seconds.
timeout /t 5 /nobreak >nul
goto :loop

:done
endlocal
