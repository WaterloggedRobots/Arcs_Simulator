@echo off
rem Build dist\ArcsDeck.exe (Steam Deck edition, one file). Run on Windows.
rem Needs Python 3.12 (pyaubo-sdk has no wheels for newer Pythons) and the
rem "py" launcher from python.org. The build venv is .venv-deck-build.
setlocal
cd /d "%~dp0..\.."

where py >nul 2>nul || (echo The "py" launcher was not found. Install Python 3.12 from python.org. & exit /b 1)
if not exist .venv-deck-build\Scripts\python.exe (
    py -3.12 -m venv .venv-deck-build || exit /b 1
)
set PY=.venv-deck-build\Scripts\python.exe
%PY% -m pip install --upgrade pip || exit /b 1
%PY% -m pip install -r packaging\deck\requirements-build.txt || exit /b 1
%PY% -m PyInstaller --noconfirm --clean --distpath dist --workpath build\deck packaging\deck\ArcsDeck.spec || exit /b 1

echo.
echo Built dist\ArcsDeck.exe
