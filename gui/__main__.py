"""
Entry point for PyInstaller
This ensures the module paths are set correctly before importing
"""
import os
import sys

# 設定模組路徑（與 run_online_control.py 相同）
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)
src_dir = os.path.join(app_dir, 'src')
sys.path.insert(0, src_dir)

# 導入並執行主程式
from run_online_control import main

if __name__ == "__main__":
    main()

