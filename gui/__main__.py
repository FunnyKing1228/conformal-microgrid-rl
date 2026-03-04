"""
Entry point for GUI / PyInstaller
啟動 P302 微電網 AI 控制介面
"""
import os
import sys

# 確保專案根目錄在路徑中
gui_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(gui_dir)
sys.path.insert(0, project_root)

from gui.ai_control_gui import main

if __name__ == "__main__":
    main()
