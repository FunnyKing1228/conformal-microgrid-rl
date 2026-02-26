@echo off
setlocal
chcp 65001 >nul

echo ========================================
echo 打包 Microgrid_AI 為獨立 EXE 檔案
echo ========================================
echo.

REM 檢查 PyInstaller 是否安裝
py -3 -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [安裝] PyInstaller 未安裝，正在安裝...
    py -3 -m pip install pyinstaller
    if errorlevel 1 (
        echo [錯誤] PyInstaller 安裝失敗
        pause
        exit /b 1
    )
)

echo [步驟 1/2] 打包 GUI 程式...
py -3 -m PyInstaller build_gui_exe_complete.spec --clean --noconfirm

if errorlevel 1 (
    echo [錯誤] GUI 打包失敗
    pause
    exit /b 1
)

echo.
echo [步驟 2/2] 打包控制程式（run_online_control.exe）...
py -3 -m PyInstaller build_control_exe.spec --clean --noconfirm

if errorlevel 1 (
    echo [警告] 控制程式打包失敗，但 GUI 已打包完成
    echo GUI 仍可使用 Python 腳本模式運行
)

echo.
echo ========================================
echo 打包完成！
echo ========================================
echo.
echo 打包模式：資料夾模式（one-folder mode）
echo 啟動時不需要解壓縮，速度更快
echo.
echo 發布目錄：release\
echo    - GUI: release\Microgrid_AI_GUI.exe
echo    - 控制程式: release\run_online_control.exe
echo    - 所有 DLL 和依賴檔案都在 release\ 目錄中
echo.
echo 使用方法：
echo   1. 將整個 release 資料夾複製到目標電腦
echo   2. 雙擊 release\Microgrid_AI_GUI.exe 即可運行
echo   3. 類似廠商軟體的方式，一點就開，不需要解壓
echo.

REM 建立發布目錄（清空舊的）
if exist "release" rmdir /s /q "release"
mkdir "release"

REM 複製資料夾模式的輸出（one-folder 模式會產生資料夾）
REM 將所有文件合併到 release 目錄中（類似廠商軟體的方式）
if exist "dist\Microgrid_AI_GUI" (
    echo 複製 GUI 檔案...
    xcopy /E /I /Y "dist\Microgrid_AI_GUI\*" "release\"
) else (
    echo [警告] 找不到 dist\Microgrid_AI_GUI 資料夾，嘗試複製單一 EXE...
    if exist "dist\Microgrid_AI_GUI.exe" copy "dist\Microgrid_AI_GUI.exe" "release\"
)

if exist "dist\run_online_control" (
    echo 複製控制程式檔案...
    xcopy /E /I /Y "dist\run_online_control\*" "release\"
) else (
    echo [警告] 找不到 dist\run_online_control 資料夾，嘗試複製單一 EXE...
    if exist "dist\run_online_control.exe" copy "dist\run_online_control.exe" "release\"
)

REM 複製配置文件（如果存在）
if exist "config_parse.json" (
    copy "config_parse.json" "release\"
    echo 已複製配置文件到 release\config_parse.json
)
if exist "PARSE_CONFIG_README.md" (
    copy "PARSE_CONFIG_README.md" "release\"
    echo 已複製配置文件說明到 release\PARSE_CONFIG_README.md
)

REM 複製 scripts 目錄（包含負載模式讀取器和 SoC/SoH 計算腳本）
if exist "scripts" (
    echo 複製 scripts 目錄...
    if not exist "release\scripts" mkdir "release\scripts"
    xcopy /E /I /Y "scripts\*.py" "release\scripts\"
    echo 已複製 scripts 目錄到 release\scripts
)

REM 複製負載模式文件（從 release 目錄或專案根目錄）
if exist "release\load_pattern.txt" (
    echo 負載模式文件已存在於 release 目錄
) else (
    if exist "load_pattern.txt" (
        copy "load_pattern.txt" "release\"
        echo 已複製負載模式文件到 release\load_pattern.txt
    ) else (
        if exist "release\load_pattern.txt" (
            echo 負載模式文件已存在
        ) else (
            echo [提示] 負載模式文件不存在，將使用預設值（負載固定為 0）
        )
    )
)

echo.
echo 發布檔案已複製到 release\ 目錄
echo 可以直接將整個 release 目錄打包傳送
echo.
echo 注意：config_parse.json 是解析配置文件，可在運行時修改，無需重新打包
echo 注意：load_pattern.txt 是負載模式文件，可在運行時修改，無需重新打包
echo.

pause
endlocal
