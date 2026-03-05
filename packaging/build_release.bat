@echo off
setlocal
chcp 65001 >nul
title P302 AI - Build Release

echo ================================================
echo   P302 Microgrid AI - Build EXE Release
echo ================================================
echo.

REM === 確認 Python ===
where python >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=python
) else if exist "C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON="C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe"
) else (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)
echo Python: %PYTHON%

REM === 確認 PyInstaller ===
%PYTHON% -c "import PyInstaller; print(f'PyInstaller {PyInstaller.__version__}')" 2>nul
if errorlevel 1 (
    echo [INFO] Installing PyInstaller...
    %PYTHON% -m pip install pyinstaller
)

REM === 切到 packaging 目錄 ===
cd /d "%~dp0"
echo Working dir: %CD%
echo.

REM === 清理舊 build ===
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

REM === 打包 ===
echo [1/2] Building P302_AI_GUI...
%PYTHON% -m PyInstaller build_release.spec --clean --noconfirm

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    echo.
    echo Common fixes:
    echo   pip install torch numpy pyyaml gymnasium pyinstaller
    echo.
    pause
    exit /b 1
)

REM === 組裝 release 資料夾 ===
echo.
echo [2/2] Assembling release folder...

set RELEASE=%~dp0..\release_exe
if exist "%RELEASE%" rmdir /s /q "%RELEASE%"
mkdir "%RELEASE%"

REM 複製 PyInstaller 產出
xcopy /E /I /Y "dist\P302_AI_GUI\*" "%RELEASE%\"

REM 複製 bat 捷徑
echo @echo off > "%RELEASE%\啟動GUI.bat"
echo cd /d "%%~dp0" >> "%RELEASE%\啟動GUI.bat"
echo start P302_AI_GUI.exe >> "%RELEASE%\啟動GUI.bat"

echo.
echo ================================================
echo   Build complete!
echo ================================================
echo.
echo Release folder: %RELEASE%
echo.
echo Usage:
echo   1. Copy the entire release_exe folder to target PC
echo   2. Double-click P302_AI_GUI.exe (or 啟動GUI.bat)
echo   3. No Python installation needed!
echo.

REM 顯示大小
for /f "tokens=3" %%a in ('dir "%RELEASE%" /s /-c ^| find "File(s)"') do set SIZE=%%a
echo Total size: %SIZE% bytes
echo.

pause
endlocal
