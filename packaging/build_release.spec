# -*- mode: python ; coding: utf-8 -*-
"""
P302 Microgrid AI — PyInstaller spec
打包 GUI + 控制腳本 + 模型 + 所有依賴為單一資料夾
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(SPECPATH))

a = Analysis(
    # 主入口
    [os.path.join(PROJECT_ROOT, 'gui', 'ai_control_gui.py')],

    pathex=[PROJECT_ROOT],

    binaries=[],

    # 附帶資料檔案：(src, dest_in_bundle)
    datas=[
        # 控制腳本（需要被子進程呼叫，也需要被 --mode 匯入）
        (os.path.join(PROJECT_ROOT, 'control', '__init__.py'),            'control'),
        (os.path.join(PROJECT_ROOT, 'control', 'solar_test_collect.py'),  'control'),
        (os.path.join(PROJECT_ROOT, 'control', 'run_deployment.py'),      'control'),
        (os.path.join(PROJECT_ROOT, 'control', 'run_online_control.py'),  'control'),
        (os.path.join(PROJECT_ROOT, 'control', 'io_protocol.py'),         'control'),

        # Core 模組（run_deployment.py 需要 import）
        (os.path.join(PROJECT_ROOT, 'core', 'sac_agent.py'),       'core'),
        (os.path.join(PROJECT_ROOT, 'core', 'microgrid_env.py'),   'core'),
        (os.path.join(PROJECT_ROOT, 'core', 'safety_net.py'),      'core'),

        # SoH predictor
        (os.path.join(PROJECT_ROOT, 'core', 'soh_predictor'),      'core/soh_predictor'),

        # 模型權重 (latest best — 50mA/100mW specs)
        (os.path.join(PROJECT_ROOT, 'models', 'best_sac_model.pth'),  'models'),

        # 設定檔
        (os.path.join(PROJECT_ROOT, 'configs', 'config_p302_sim.yaml'),  'configs'),

        # 訓練資料（參考用）
        (os.path.join(PROJECT_ROOT, 'data', 'processed', 'training_7day_15min.csv'),
         'data/processed'),
    ],

    hiddenimports=[
        'tkinter', 'tkinter.ttk', 'tkinter.messagebox',
        'tkinter.filedialog', 'tkinter.scrolledtext',
        'torch', 'numpy', 'yaml',
        'csv', 'json', 'argparse',
        # core modules that deployment script imports
        'core.sac_agent', 'core.microgrid_env', 'core.safety_net',
        'core.soh_predictor', 'core.soh_predictor.inference',
        # control modules（--mode dispatch 需要匯入）
        'control', 'control.io_protocol',
        'control.solar_test_collect', 'control.run_deployment',
    ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 繪圖/科學計算（部署不需要）
        'matplotlib', 'scipy', 'pandas', 'PIL', 'Pillow',
        'plotly', 'bokeh', 'altair', 'seaborn', 'holoviews',
        'skimage', 'sklearn', 'scikit-learn', 'scikit-image',
        # Jupyter/IPython
        'IPython', 'notebook', 'jupyter', 'jupyter_core',
        'jupyter_client', 'jupyter_server', 'jupyterlab',
        'nbconvert', 'nbformat', 'ipykernel', 'ipywidgets',
        # Torch 擴展
        'torchaudio', 'torchvision', 'torchtext',
        # Qt（我們用 tkinter）
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'sip',
        # Sphinx/文件
        'sphinx', 'sphinxcontrib', 'docutils', 'alabaster',
        # 其他不需要的
        'lxml', 'pygments', 'pytest', 'setuptools', '_pytest',
        'pyviz_comms', 'panel', 'param', 'intake', 'dask',
        'distributed', 'fsspec', 'cytoolz', 'toolz',
        'numba', 'llvmlite', 'cffi', 'cryptography', 'bcrypt',
        'zmq', 'tornado', 'jinja2', 'markupsafe',
        'chardet', 'charset_normalizer', 'certifi', 'urllib3',
        'requests', 'httpx', 'aiohttp',
        'h5py', 'tables', 'xlrd', 'openpyxl',
        'sympy', 'astropy', 'statsmodels',
        'conda', 'anaconda_navigator',
        'gymnasium', 'gym', 'pygame', 'box2d',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='P302_AI_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI 不需要控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='P302_AI_GUI',
)
