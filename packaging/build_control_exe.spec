# -*- mode: python ; coding: utf-8 -*-
"""
run_online_control.py 的 EXE 打包配置
"""

import os
from PyInstaller.utils.hooks import collect_submodules

# 取得專案根目錄（spec 檔案所在目錄）
# PyInstaller 執行時，工作目錄就是 spec 檔案所在目錄
project_root = os.getcwd()

# 收集所有 app 模組
app_modules = collect_submodules('app')

# 隱藏導入
hiddenimports = [
    'numpy',
    'torch',
    'argparse',
    'datetime',
    'time',
    'math',
    'os',
    'sys',
    # app 模組
    'app',
    'app.io_protocol',
    'app.data_collector',
    'app.src',
    'app.src.sac_agent',
    'app.src.safety_net',
    # scripts 模組
    'scripts',
    'scripts.load_pattern_reader',
    'scripts.calculate_soc_soh',
]

# 收集 scripts 目錄的資料檔案
datas = []
scripts_dir = os.path.join(project_root, 'scripts')
if os.path.exists(scripts_dir):
    # 包含 scripts 目錄中的所有 .py 文件
    for root, dirs, files in os.walk(scripts_dir):
        for file in files:
            if file.endswith('.py'):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, project_root)
                datas.append((src_path, os.path.dirname(rel_path)))

a = Analysis(
    [os.path.join(project_root, 'app', 'run_online_control.py')],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# 改為 one-folder 模式（資料夾模式）：啟動時不需要解壓，速度更快
exe = EXE(
    pyz,
    a.scripts,
    [],  # 不包含 binaries、zipfiles、datas（這些會放在資料夾中）
    exclude_binaries=True,  # 關鍵：不打包到 EXE 中
    name='run_online_control',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # 關閉 UPX 壓縮以提高啟動速度
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 控制台程式需要顯示輸出
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 收集所有文件到資料夾（one-folder 模式的關鍵）
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='run_online_control',  # 資料夾名稱
)

