#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 控制程式 GUI 介面 - 測試場景模式

功能：
1. AI 控制模式（使用 AI 模型）
2. 數據傳輸測試（生成測試數據）
3. ID=0 電池不控制測試
"""

import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
from datetime import datetime

# 設定檔案路徑
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config_gui.json")


class ConfigManager:
    """管理設定檔"""
    
    @staticmethod
    def load():
        """載入設定"""
        defaults = {
            "vendor_dir": r"C:\Users\Administrator\Downloads\P302_AI_v2.5",
            "vendor_exe": r"C:\Users\Administrator\Downloads\P302_AI_v2.5\P302.exe",
            "model_path": os.path.join(os.path.dirname(__file__), "..", "models", "best_sac_model.pth"),
            "initial_soc": 50.0,
            "max_power_w": 50.0,  # 改為 W（瓦特），去掉 k 單位
            "max_flow_percent": 100.0,
            "battery_count": 1,  # 本次實驗使用的電池數量（預設 1 顆）
            "collect_data": False,
            "data_output_dir": "",
            "collect_interval_sec": 900,
            "use_watchdog": True,  # 預設啟用 Watchdog 自動重啟
            "watchdog_check_interval_sec": 60,  # Watchdog 檢查間隔（秒）
        }
        
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    defaults.update(config)
            except Exception:
                pass
        
        return defaults
    
    @staticmethod
    def save(config):
        """儲存設定"""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False


def check_vendor_running(vendor_exe_path: str) -> bool:
    """檢查廠商程式是否有在執行（非阻塞，快速檢查）"""
    try:
        exe_name = os.path.basename(vendor_exe_path).lower()
        # 使用 timeout 避免卡住
        out = subprocess.check_output(
            ["tasklist"], 
            encoding="utf-8", 
            errors="ignore",
            timeout=2,  # 2 秒超時
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        )
        return exe_name in out.lower()
    except (subprocess.TimeoutExpired, Exception):
        return False


class ScriptProcess:
    """管理腳本執行流程，支援即時日誌輸出"""
    
    def __init__(self, name: str, cmd: list, log_callback=None):
        self.name = name
        self.cmd = cmd
        self.proc: subprocess.Popen | None = None
        self.log_callback = log_callback
        self.log_thread: threading.Thread | None = None
        self._stop_logging = False
    
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None
    
    def _read_output(self, pipe):
        """讀取輸出並透過 callback 顯示"""
        try:
            while not self._stop_logging:
                line = pipe.readline()
                if not line:
                    # 如果進程已經結束，嘗試讀取剩餘輸出
                    if self.proc and self.proc.poll() is not None:
                        # 讀取所有剩餘輸出
                        try:
                            remaining = pipe.read()
                            if remaining:
                                if isinstance(remaining, bytes):
                                    text = remaining.decode('utf-8', errors='replace')
                                else:
                                    text = remaining
                                if text and self.log_callback:
                                    self.log_callback(f"{text}\n")
                        except Exception:
                            pass
                    break
                try:
                    if isinstance(line, bytes):
                        text = line.decode('utf-8', errors='replace').rstrip()
                    else:
                        text = line.rstrip()
                    if text and self.log_callback:
                        self.log_callback(f"{text}\n")
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass
            # 如果進程已經結束，記錄退出碼
            if self.proc and self.proc.poll() is not None:
                exit_code = self.proc.returncode
                if self.log_callback and exit_code != 0:
                    self.log_callback(f"[錯誤] 進程異常退出，退出碼: {exit_code}\n")
    
    def start(self, cwd: str = None):
        if self.is_running():
            return False
        try:
            self._stop_logging = False
            # 記錄啟動資訊
            if self.log_callback:
                self.log_callback(f"[DEBUG] 啟動進程: {' '.join(self.cmd)}\n")
                self.log_callback(f"[DEBUG] 工作目錄: {cwd}\n")
            
            self.proc = subprocess.Popen(
                self.cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                bufsize=0,
                text=True,  # 確保輸出是文字模式
                encoding='utf-8',
                errors='replace',
            )
            
            # 啟動日誌讀取線程
            self.log_thread = threading.Thread(
                target=self._read_output,
                args=(self.proc.stdout,),
                daemon=True
            )
            self.log_thread.start()
            
            if self.log_callback:
                self.log_callback(f"[DEBUG] 進程已啟動，PID: {self.proc.pid}\n")
            
            # 等待一小段時間，檢查進程是否立即崩潰
            time.sleep(0.5)
            if self.proc.poll() is not None:
                # 進程已經結束，讀取剩餘輸出
                exit_code = self.proc.returncode
                if self.log_callback:
                    self.log_callback(f"[錯誤] 進程立即退出，退出碼: {exit_code}\n")
                    self.log_callback(f"[錯誤] 請檢查上面的錯誤訊息\n")
                return False
            
            return True
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"[錯誤] 啟動失敗: {e}\n")
                import traceback
                self.log_callback(f"[錯誤] 詳細錯誤: {traceback.format_exc()}\n")
            return False
    
    def stop(self):
        if not self.is_running():
            return
        self._stop_logging = True
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        except Exception:
            pass
        finally:
            if self.log_callback:
                self.log_callback(f"[{datetime.now().strftime('%H:%M:%S')}] [{self.name}] 已停止\n")
            self.proc = None


class AIControlGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI 控制程式管理介面 - 測試場景")
        self.geometry("1400x800")  # 調整為更寬的視窗，適合左右布局
        
        # 載入設定
        self.config = ConfigManager.load()
        
        # 狀態變數
        self.vendor_status_var = tk.StringVar(value="Unknown")
        self.scenario_status_var = tk.StringVar(value="已停止")
        
        # 當前執行的場景
        self.current_scenario: ScriptProcess | None = None
        
        # 廠商程式進程（如果由 GUI 啟動）
        self.vendor_proc: subprocess.Popen | None = None
        
        # Watchdog 監控（內建功能）
        self.watchdog_thread: threading.Thread | None = None
        self.watchdog_stop_flag = threading.Event()
        self.watchdog_cmd_backup: list = []  # 備份啟動命令，用於自動重啟
        self.watchdog_project_root: str = ""  # 備份工作目錄
        self.watchdog_restart_count = 0  # 重啟次數計數
        
        # Data/Command 狀態（用於分頁顯示）
        self.last_data_content = ""
        self.last_command_content = ""
        
        # 建立 UI
        self._build_ui()
        
        # 綁定窗口關閉事件
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # 延遲啟動狀態檢查（避免啟動時卡頓）
        # 先顯示 GUI，再執行耗時操作
        self.after(50, self._update_vendor_status)  # 50ms 後檢查（更快）
        self.after(150, self._poll_status)  # 150ms 後開始狀態輪詢
        self.after(300, self._poll_data_command)  # 300ms 後開始檔案輪詢
        
        # 清理殘留進程延遲到更後面（不影響 GUI 顯示）
        self.after(1000, self._cleanup_orphaned_processes)  # 1 秒後清理
    
    def _build_ui(self):
        pad = 8
        
        # === 主布局：左右分割 ===
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill="both", expand=True, padx=pad, pady=pad)
        
        # === 左側：基本設定 + 日誌/狀態顯示 ===
        frame_left_container = ttk.Frame(main_paned)
        main_paned.add(frame_left_container, weight=1)
        
        # 左側上半部：基本設定
        frame_left = ttk.LabelFrame(frame_left_container, text="基本設定")
        frame_left.pack(fill="x", padx=pad, pady=(pad, pad//2))
        
        # 廠商程式位置
        row1 = ttk.Frame(frame_left)
        row1.pack(fill="x", padx=pad, pady=4)
        ttk.Label(row1, text="廠商程式資料夾:").pack(side="left", padx=4)
        self.vendor_dir_var = tk.StringVar(value=self.config.get("vendor_dir", ""))
        ttk.Entry(row1, textvariable=self.vendor_dir_var, width=40).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row1, text="瀏覽", command=self._browse_vendor_dir).pack(side="left", padx=4)
        
        # 廠商程式 exe
        row2 = ttk.Frame(frame_left)
        row2.pack(fill="x", padx=pad, pady=4)
        ttk.Label(row2, text="廠商程式 EXE:  ").pack(side="left", padx=4)
        self.vendor_exe_var = tk.StringVar(value=self.config.get("vendor_exe", ""))
        ttk.Entry(row2, textvariable=self.vendor_exe_var, width=40).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row2, text="瀏覽", command=self._browse_vendor_exe).pack(side="left", padx=4)
        
        # 廠商程式狀態
        row_status = ttk.Frame(frame_left)
        row_status.pack(fill="x", padx=pad, pady=8)
        ttk.Label(row_status, text="廠商程式狀態:").pack(side="left", padx=4)
        ttk.Label(row_status, textvariable=self.vendor_status_var, width=15).pack(side="left", padx=4)
        ttk.Button(row_status, text="重新檢查", command=self._update_vendor_status).pack(side="left", padx=4)
        ttk.Button(row_status, text="啟動廠商程式", command=self._launch_vendor).pack(side="left", padx=4)
        
        # 儲存設定按鈕
        ttk.Button(frame_left, text="儲存設定", command=self._save_config).pack(pady=4)
        
        # 左側下半部：日誌和狀態顯示
        frame_left_bottom = ttk.LabelFrame(frame_left_container, text="資訊顯示")
        frame_left_bottom.pack(fill="both", expand=True, padx=pad, pady=(pad//2, pad))
        
        # 使用分頁顯示日誌和 Data/Command
        notebook_left = ttk.Notebook(frame_left_bottom)
        notebook_left.pack(fill="both", expand=True, padx=pad, pady=pad)
        
        # 分頁1：執行日誌
        frame_log = ttk.Frame(notebook_left)
        notebook_left.add(frame_log, text="執行日誌")
        self.log_text = scrolledtext.ScrolledText(frame_log, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=pad, pady=pad)
        self.log_text.config(state=tk.DISABLED)
        
        # 分頁2：Data/Command 狀態
        frame_status = ttk.Frame(notebook_left)
        notebook_left.add(frame_status, text="Data/Command")
        
        # 使用上下分割顯示 Data 和 Command
        paned_status = ttk.PanedWindow(frame_status, orient=tk.VERTICAL)
        paned_status.pack(fill="both", expand=True, padx=pad, pady=pad)
        
        # Data 顯示區（上半部）
        frame_data = ttk.LabelFrame(paned_status, text="收到的 Data.txt")
        paned_status.add(frame_data, weight=1)
        self.data_text = scrolledtext.ScrolledText(frame_data, wrap=tk.NONE, font=("Consolas", 9))
        self.data_text.pack(fill="both", expand=True, padx=pad, pady=pad)
        self.data_text.config(state=tk.DISABLED)
        
        # Command 顯示區（下半部）
        frame_cmd = ttk.LabelFrame(paned_status, text="輸出的 Command.txt")
        paned_status.add(frame_cmd, weight=1)
        self.cmd_text = scrolledtext.ScrolledText(frame_cmd, wrap=tk.NONE, font=("Consolas", 9))
        self.cmd_text.pack(fill="both", expand=True, padx=pad, pady=pad)
        self.cmd_text.config(state=tk.DISABLED)
        
        # === 右側：測試場景設定（可滾動） ===
        frame_right = ttk.LabelFrame(main_paned, text="測試場景")
        main_paned.add(frame_right, weight=1)
        
        # 創建可滾動的框架
        # 使用 Canvas + Scrollbar 實現滾動功能
        canvas = tk.Canvas(frame_right, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame_right, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # 配置滾動區域
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        
        # 將可滾動框架放入 Canvas
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # 確保 Canvas 寬度與內容同步
        def configure_canvas_width(event):
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        canvas.bind("<Configure>", configure_canvas_width)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 綁定滑鼠滾輪事件
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # 布局 Canvas 和 Scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 場景說明
        desc_text = """選擇測試場景：

1. AI 控制
   - 使用 AI 模型推論
   - 需要模型檔案

2. 數據傳輸測試
   - 生成測試數據
   - 不需模型，測試通信

3. ID=0 電池不控制測試
   - 測試 ID=0 情況
   - 輸出 0 指令
"""
        ttk.Label(scrollable_frame, text=desc_text, justify="left", font=("", 9)).pack(anchor="w", padx=pad, pady=pad)
        
        # 場景選擇
        self.scenario_var = tk.StringVar(value="ai")
        ttk.Radiobutton(scrollable_frame, text="1. AI 控制", variable=self.scenario_var, 
                       value="ai").pack(anchor="w", padx=pad, pady=2)
        ttk.Radiobutton(scrollable_frame, text="2. 數據傳輸測試", variable=self.scenario_var, 
                       value="test").pack(anchor="w", padx=pad, pady=2)
        ttk.Radiobutton(scrollable_frame, text="3. ID=0 電池不控制測試", variable=self.scenario_var, 
                       value="id0").pack(anchor="w", padx=pad, pady=2)
        
        # 電池規格設定
        frame_battery_spec = ttk.LabelFrame(scrollable_frame, text="電池規格設定（必填）")
        frame_battery_spec.pack(fill="x", padx=pad, pady=4)
        
        # 初始 SoC
        row_soc = ttk.Frame(frame_battery_spec)
        row_soc.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_soc, text="初始 SoC (%):").pack(side="left", padx=4)
        self.initial_soc_var = tk.StringVar(value=str(self.config.get("initial_soc", 50.0)))
        ttk.Entry(row_soc, textvariable=self.initial_soc_var, width=15).pack(side="left", padx=4)
        ttk.Label(row_soc, text="(當收到的 SoC 為 0 時使用)").pack(side="left", padx=4)
        
        # 最大功率（改為 W，去掉 k 單位）
        row_pmax = ttk.Frame(frame_battery_spec)
        row_pmax.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_pmax, text="最大功率 (W):").pack(side="left", padx=4)
        self.max_power_var = tk.StringVar(value=str(self.config.get("max_power_w", 50.0)))
        ttk.Entry(row_pmax, textvariable=self.max_power_var, width=15).pack(side="left", padx=4)
        hint_label = ttk.Label(row_pmax, text="(建議最小值: 0.1W)")
        hint_label.pack(side="left", padx=4)
        
        # 最大流速
        row_flow = ttk.Frame(frame_battery_spec)
        row_flow.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_flow, text="最大流速 (%):").pack(side="left", padx=4)
        self.max_flow_var = tk.StringVar(value=str(self.config.get("max_flow_percent", 100.0)))
        ttk.Entry(row_flow, textvariable=self.max_flow_var, width=15).pack(side="left", padx=4)
        
        # 電池數量（資料收集用）
        row_count = ttk.Frame(frame_battery_spec)
        row_count.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_count, text="電池數量:").pack(side="left", padx=4)
        self.battery_count_var = tk.StringVar(value=str(self.config.get("battery_count", 1)))
        ttk.Entry(row_count, textvariable=self.battery_count_var, width=15).pack(side="left", padx=4)
        ttk.Label(row_count, text="(資料收集時只記錄前 N 顆電池，預設 1)").pack(side="left", padx=4)
        
        # 電池容量（用於理論計算 SoC，改為 Wh，去掉 k 單位）
        row_capacity = ttk.Frame(frame_battery_spec)
        row_capacity.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_capacity, text="電池容量 (Wh):").pack(side="left", padx=4)
        self.battery_capacity_var = tk.StringVar(value=str(self.config.get("battery_capacity_wh", 10.0)))
        ttk.Entry(row_capacity, textvariable=self.battery_capacity_var, width=15).pack(side="left", padx=4)
        ttk.Label(row_capacity, text="(用於理論計算 SoC 變化，預設 10.0)").pack(side="left", padx=4)
        
        # 電池效率（用於理論計算 SoC）
        row_efficiency = ttk.Frame(frame_battery_spec)
        row_efficiency.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_efficiency, text="電池效率 (0-1):").pack(side="left", padx=4)
        self.battery_efficiency_var = tk.StringVar(value=str(self.config.get("battery_efficiency", 0.95)))
        ttk.Entry(row_efficiency, textvariable=self.battery_efficiency_var, width=15).pack(side="left", padx=4)
        ttk.Label(row_efficiency, text="(用於理論計算 SoC 變化，預設 0.95 = 95%)").pack(side="left", padx=4)
        
        # 負載規格設定
        frame_load_spec = ttk.LabelFrame(scrollable_frame, text="負載規格設定")
        frame_load_spec.pack(fill="x", padx=pad, pady=4)
        
        # 每顆負載功率
        row_load_power = ttk.Frame(frame_load_spec)
        row_load_power.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_load_power, text="每顆負載功率 (W):").pack(side="left", padx=4)
        self.load_power_per_unit_var = tk.StringVar(value=str(self.config.get("load_power_per_unit_w", 1000.0)))
        ttk.Entry(row_load_power, textvariable=self.load_power_per_unit_var, width=15).pack(side="left", padx=4)
        ttk.Label(row_load_power, text="(每顆負載的功率，例如 1000W = 1kW)").pack(side="left", padx=4)
        
        # 最大負載顆數
        row_max_load = ttk.Frame(frame_load_spec)
        row_max_load.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_max_load, text="最大負載顆數:").pack(side="left", padx=4)
        self.max_load_count_var = tk.StringVar(value=str(self.config.get("max_load_count", 10)))
        ttk.Entry(row_max_load, textvariable=self.max_load_count_var, width=15).pack(side="left", padx=4)
        ttk.Label(row_max_load, text="(負載顆數上限，預設 10)").pack(side="left", padx=4)
        
        # 負載模式文件
        row_load_pattern = ttk.Frame(frame_load_spec)
        row_load_pattern.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_load_pattern, text="負載模式文件:").pack(side="left", padx=4)
        self.load_pattern_file_var = tk.StringVar(value=self.config.get("load_pattern_file", ""))
        ttk.Entry(row_load_pattern, textvariable=self.load_pattern_file_var, width=30).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row_load_pattern, text="瀏覽", command=self._browse_load_pattern_file).pack(side="left", padx=4)
        
        # 模型路徑（僅 AI 模式需要）
        row_model = ttk.Frame(scrollable_frame)
        row_model.pack(fill="x", padx=pad, pady=4)
        ttk.Label(row_model, text="模型檔案:").pack(side="left", padx=4)
        self.model_path_var = tk.StringVar(value=self.config.get("model_path", ""))
        ttk.Entry(row_model, textvariable=self.model_path_var, width=30).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row_model, text="瀏覽", command=self._browse_model).pack(side="left", padx=4)
        
        # 資料收集設定
        frame_collect = ttk.LabelFrame(scrollable_frame, text="資料收集設定")
        frame_collect.pack(fill="x", padx=pad, pady=4)
        
        self.collect_data_var = tk.BooleanVar(value=self.config.get("collect_data", False))
        ttk.Checkbutton(frame_collect, text="啟用資料收集", variable=self.collect_data_var).pack(anchor="w", padx=pad, pady=2)
        
        row_output = ttk.Frame(frame_collect)
        row_output.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_output, text="輸出目錄:").pack(side="left", padx=4)
        self.data_output_dir_var = tk.StringVar(value=self.config.get("data_output_dir", ""))
        ttk.Entry(row_output, textvariable=self.data_output_dir_var, width=30).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row_output, text="瀏覽", command=self._browse_data_output_dir).pack(side="left", padx=4)
        
        row_interval = ttk.Frame(frame_collect)
        row_interval.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_interval, text="收集間隔 (秒):").pack(side="left", padx=4)
        self.collect_interval_var = tk.StringVar(value=str(self.config.get("collect_interval_sec", 900)))
        ttk.Entry(row_interval, textvariable=self.collect_interval_var, width=15).pack(side="left", padx=4)
        ttk.Label(row_interval, text="(預設 900 = 15 分鐘)").pack(side="left", padx=4)
        
        # Watchdog 自動重啟設定（內建功能）
        self.use_watchdog_var = tk.BooleanVar(value=self.config.get("use_watchdog", True))
        ttk.Checkbutton(
            frame_collect, 
            text="啟用 Watchdog 自動重啟（推薦）", 
            variable=self.use_watchdog_var
        ).pack(anchor="w", padx=pad, pady=2)
        
        row_watchdog_interval = ttk.Frame(frame_collect)
        row_watchdog_interval.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_watchdog_interval, text="Watchdog 檢查間隔 (秒):").pack(side="left", padx=4)
        self.watchdog_check_interval_var = tk.StringVar(value=str(self.config.get("watchdog_check_interval_sec", 60)))
        ttk.Entry(row_watchdog_interval, textvariable=self.watchdog_check_interval_var, width=15).pack(side="left", padx=4)
        ttk.Label(row_watchdog_interval, text="(預設 60 = 1 分鐘，檢測到停止後會自動重啟)").pack(side="left", padx=4)
        
        # SoC/SoH 計算設定
        frame_soc_soh = ttk.LabelFrame(scrollable_frame, text="SoC/SoH 計算設定")
        frame_soc_soh.pack(fill="x", padx=pad, pady=4)
        
        self.enable_soc_soh_var = tk.BooleanVar(value=self.config.get("enable_soc_soh", False))
        ttk.Checkbutton(frame_soc_soh, text="啟用 SoC/SoH 計算（根據 Command.txt 計算）", variable=self.enable_soc_soh_var).pack(anchor="w", padx=pad, pady=2)
        
        row_soc_soh_output = ttk.Frame(frame_soc_soh)
        row_soc_soh_output.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_soc_soh_output, text="SoC/SoH 輸出 CSV:").pack(side="left", padx=4)
        self.soc_soh_output_csv_var = tk.StringVar(value=self.config.get("soc_soh_output_csv", ""))
        ttk.Entry(row_soc_soh_output, textvariable=self.soc_soh_output_csv_var, width=30).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row_soc_soh_output, text="瀏覽", command=self._browse_soc_soh_output).pack(side="left", padx=4)
        
        row_soc_soh_interval = ttk.Frame(frame_soc_soh)
        row_soc_soh_interval.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_soc_soh_interval, text="讀取間隔 (秒):").pack(side="left", padx=4)
        self.soc_soh_read_interval_var = tk.StringVar(value=str(self.config.get("soc_soh_read_interval_sec", 1.0)))
        ttk.Entry(row_soc_soh_interval, textvariable=self.soc_soh_read_interval_var, width=15).pack(side="left", padx=4)
        ttk.Label(row_soc_soh_interval, text="(預設 1.0 = 每秒讀取 Command.txt)").pack(side="left", padx=4)
        
        # 場景狀態和控制按鈕
        status_frame = ttk.Frame(scrollable_frame)
        status_frame.pack(fill="x", padx=pad, pady=pad)
        ttk.Label(status_frame, text="狀態:").pack(side="left", padx=4)
        ttk.Label(status_frame, textvariable=self.scenario_status_var, width=15).pack(side="left", padx=4)
        
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill="x", padx=pad, pady=4)
        ttk.Button(btn_frame, text="啟動場景", command=self._start_scenario).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="停止場景", command=self._stop_scenario).pack(side="left", padx=4)
        
        # 底部按鈕（放在右側底部）
        frame_bottom = ttk.Frame(frame_right)
        frame_bottom.pack(fill="x", padx=pad, pady=pad)
        ttk.Button(frame_bottom, text="開啟廠商資料夾", command=self._open_vendor_folder).pack(side="left", padx=4)
        ttk.Button(frame_bottom, text="清除日誌", command=self._clear_log).pack(side="left", padx=4)
    
    def _log(self, message: str):
        """添加日誌訊息，並解析 Data/Command 內容"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # 解析 Data 和 Command 內容
        self._parse_log_for_status(message)
    
    def _clear_log(self):
        """清除日誌"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.data_text.config(state=tk.NORMAL)
        self.data_text.delete(1.0, tk.END)
        self.data_text.config(state=tk.DISABLED)
        
        self.cmd_text.config(state=tk.NORMAL)
        self.cmd_text.delete(1.0, tk.END)
        self.cmd_text.config(state=tk.DISABLED)
    
    def _parse_log_for_status(self, message: str):
        """從日誌訊息中解析 Data 和 Command 內容"""
        lines = message.split('\n')
        for line in lines:
            line = line.strip()
            if '[Data]' in line or 'Data.txt' in line:
                # 嘗試讀取實際的 Data.txt 檔案
                vendor_dir = self.vendor_dir_var.get()
                data_file = os.path.join(vendor_dir, "Data.txt")
                if os.path.exists(data_file):
                    try:
                        with open(data_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                self.last_data_content = content
                                self._update_data_display()
                    except Exception:
                        pass
            
            if '[Command]' in line or 'Command.txt' in line or ',202' in line:
                # 解析 Command 行（格式：PP,YYYYMMDDhhmmss,...）
                if ',' in line and len(line) > 20:
                    # 可能是 Command 行
                    self.last_command_content = line
                    self._update_command_display()
    
    def _update_data_display(self):
        """更新 Data 顯示"""
        self.data_text.config(state=tk.NORMAL)
        self.data_text.delete(1.0, tk.END)
        if self.last_data_content:
            self.data_text.insert(tk.END, self.last_data_content)
        else:
            self.data_text.insert(tk.END, "(尚未讀取到 Data.txt)")
        self.data_text.see(tk.END)
        self.data_text.config(state=tk.DISABLED)
    
    def _update_command_display_from_file(self, cmd_file: str = None):
        """更新 Command 顯示（從檔案讀取）"""
        if cmd_file is None:
            vendor_dir = self.vendor_dir_var.get()
            cmd_file = os.path.join(vendor_dir, "Command.txt")
        
        if not os.path.exists(cmd_file):
            self.cmd_text.config(state=tk.NORMAL)
            self.cmd_text.delete(1.0, tk.END)
            self.cmd_text.insert(tk.END, "(Command.txt 不存在)")
            self.cmd_text.config(state=tk.DISABLED)
            return
        
        try:
            with open(cmd_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                self.cmd_text.config(state=tk.NORMAL)
                self.cmd_text.delete(1.0, tk.END)
                
                if not content:
                    self.cmd_text.insert(tk.END, "(Command.txt 為空)")
                    self.cmd_text.config(state=tk.DISABLED)
                    return
                
                # 格式化顯示：每行一個電池指令
                lines = content.split('\n')
                now_str = datetime.now().strftime("%H:%M:%S")
                self.cmd_text.insert(tk.END, f"[{now_str}] Command.txt 內容:\n")
                self.cmd_text.insert(tk.END, "=" * 80 + "\n")
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析並格式化顯示
                    parts = line.split(',')
                    if len(parts) >= 3:
                        battery_id = parts[0].strip()
                        power_str = parts[1].strip() if len(parts) > 1 else "0"  # 功率是第二個欄位
                        flow_str = parts[2].strip() if len(parts) > 2 else "0"   # 流速是第三個欄位
                        
                        # 解析功率（Command.txt 中功率是 mW，毫瓦）
                        try:
                            power_mw = int(power_str)  # 解析為整數（mW）
                            power_w = power_mw / 1000.0  # mW → W
                            
                            # 解析流速（百分比，整數）
                            flow_percent = float(flow_str) if flow_str.isdigit() else 0.0
                            
                            # 格式化顯示（去掉 kW，只顯示 W 和 mW）
                            if power_mw == 0:
                                self.cmd_text.insert(tk.END, f"電池 {battery_id}: 功率=     0.0W, 流速={flow_percent:5.1f}%\n")
                            else:
                                self.cmd_text.insert(tk.END, f"電池 {battery_id}: 功率={power_w:7.1f}W ({power_mw:8d}mW), 流速={flow_percent:5.1f}%\n")
                        except (ValueError, IndexError):
                            self.cmd_text.insert(tk.END, f"{line}\n")
                    else:
                        # 可能是時間戳行
                        if len(line) == 14 and line.isdigit():
                            self.cmd_text.insert(tk.END, f"時間戳: {line}\n")
                        else:
                            self.cmd_text.insert(tk.END, f"{line}\n")
                
                self.cmd_text.see(tk.END)
                self.cmd_text.config(state=tk.DISABLED)
        except Exception as e:
            self.cmd_text.config(state=tk.NORMAL)
            self.cmd_text.delete(1.0, tk.END)
            self.cmd_text.insert(tk.END, f"讀取錯誤: {e}")
            self.cmd_text.config(state=tk.DISABLED)
    
    def _update_command_display(self):
        """更新 Command 顯示（舊方法，保留以向後相容）"""
        self._update_command_display_from_file()
    
    def _start_scenario(self):
        """啟動選定的測試場景"""
        if self.current_scenario and self.current_scenario.is_running():
            messagebox.showwarning("警告", "已有場景在執行中，請先停止")
            return
        
        # 驗證必填欄位
        try:
            initial_soc = float(self.initial_soc_var.get())
            max_power_w = float(self.max_power_var.get())
            max_flow_percent = float(self.max_flow_var.get())
            battery_count = int(self.battery_count_var.get()) if self.battery_count_var.get() else 1
            
            if initial_soc < 0 or initial_soc > 100:
                messagebox.showerror("錯誤", "初始 SoC 必須在 0-100 之間")
                return
            if max_power_w < 0.1:
                messagebox.showerror("錯誤", f"最大功率必須至少 0.1W（您輸入的是 {max_power_w}W）\n過小的值可能導致程式卡死")
                return
            if max_power_w <= 0:
                messagebox.showerror("錯誤", "最大功率必須大於 0")
                return
            if max_flow_percent < 0 or max_flow_percent > 100:
                messagebox.showerror("錯誤", "最大流速必須在 0-100 之間")
                return
            if battery_count < 1 or battery_count > 10:
                messagebox.showerror("錯誤", "電池數量必須在 1-10 之間")
                return
            
            # 驗證 SoC/SoH 計算設定（如果啟用）
            if self.enable_soc_soh_var.get():
                if not self.soc_soh_output_csv_var.get():
                    messagebox.showerror("錯誤", "啟用 SoC/SoH 計算時，必須指定輸出 CSV 檔案路徑")
                    return
                battery_capacity_wh = float(self.battery_capacity_var.get()) if self.battery_capacity_var.get() else 0.0
                if battery_capacity_wh <= 0:
                    messagebox.showerror("錯誤", "啟用 SoC/SoH 計算時，電池容量必須大於 0")
                    return
                battery_efficiency = float(self.battery_efficiency_var.get()) if self.battery_efficiency_var.get() else 0.0
                if battery_efficiency <= 0 or battery_efficiency > 1:
                    messagebox.showerror("錯誤", "啟用 SoC/SoH 計算時，電池效率必須在 0-1 之間")
                    return
                if initial_soc < 0 or initial_soc > 100:
                    messagebox.showerror("錯誤", "啟用 SoC/SoH 計算時，初始 SoC 必須在 0-100 之間")
                    return
            
            # 驗證負載規格設定
            load_power_per_unit_w = float(self.load_power_per_unit_var.get()) if self.load_power_per_unit_var.get() else 1000.0
            max_load_count = int(self.max_load_count_var.get()) if self.max_load_count_var.get() else 10
            
            if load_power_per_unit_w <= 0:
                messagebox.showerror("錯誤", "每顆負載功率必須大於 0")
                return
            if max_load_count < 0 or max_load_count > 100:
                messagebox.showerror("錯誤", "最大負載顆數必須在 0-100 之間")
                return
        except ValueError:
            messagebox.showerror("錯誤", "請檢查電池規格設定的數值格式")
            return
        
        scenario = self.scenario_var.get()
        
        # 檢查必要設定
        vendor_dir = self.vendor_dir_var.get()
        
        status_file = os.path.join(vendor_dir, "Data.txt")
        command_file = os.path.join(vendor_dir, "Command.txt")
        
        # 根據場景選擇模式和電池 ID
        if scenario == "ai":
            battery_id = "01"
            scenario_name = "AI控制"
            test_mode = "ai"
            # 檢查模型檔案
            model_path = self.model_path_var.get()
            if not model_path or not os.path.exists(model_path):
                messagebox.showerror("錯誤", f"AI控制模式需要模型檔案: {model_path}")
                return
        elif scenario == "test":
            battery_id = "01"
            scenario_name = "數據傳輸測試"
            test_mode = "test"
            model_path = None
        elif scenario == "id0":
            battery_id = "0"
            scenario_name = "ID=0不控制測試"
            test_mode = "id0"
            model_path = None
        else:
            battery_id = "01"
            scenario_name = "測試"
            test_mode = "test"
            model_path = None
        
        # 建立命令 - 直接使用 run_online_control.py
        # 檢查是否有 EXE 檔案（打包後使用），否則使用 Python 腳本
        if getattr(sys, 'frozen', False):
            # 打包成 EXE 後，使用相對路徑找 run_online_control.exe
            exe_dir = os.path.dirname(sys.executable)
            control_exe = os.path.join(exe_dir, "run_online_control.exe")
            if os.path.exists(control_exe):
                py_cmd = [control_exe]
                project_root = exe_dir  # EXE 模式下，工作目錄就是 EXE 所在目錄
                self._log(f"[DEBUG] EXE 模式：找到 run_online_control.exe: {control_exe}\n")
                self._log(f"[DEBUG] 工作目錄: {project_root}\n")
            else:
                # 如果找不到 EXE，記錄錯誤
                self._log(f"[錯誤] 找不到 run_online_control.exe: {control_exe}\n")
                self._log(f"[錯誤] EXE 目錄: {exe_dir}\n")
                messagebox.showerror("錯誤", f"找不到 run_online_control.exe\n請確認兩個 EXE 檔案在同一目錄\n\nEXE 目錄: {exe_dir}")
                return
        else:
            # 開發模式，使用 Python 腳本
            app_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(app_dir)
            script_path = os.path.join(app_dir, "run_online_control.py")
            py_cmd = [sys.executable, script_path]
            self._log(f"[DEBUG] Python 模式：使用腳本: {script_path}\n")
            self._log(f"[DEBUG] 工作目錄: {project_root}\n")
        
        # 取得電池規格設定（單位已改為 W）
        initial_soc = float(self.initial_soc_var.get())
        max_power_w = float(self.max_power_var.get())
        max_flow_percent = float(self.max_flow_var.get())
        
        # 確保路徑使用正確的格式（Windows 使用反斜線）
        status_file_normalized = os.path.normpath(status_file)
        command_file_normalized = os.path.normpath(command_file)
        
        # 轉換為 kW 傳遞給 run_online_control.py（內部仍使用 kW）
        max_power_kw = max_power_w / 1000.0
        
        cmd = py_cmd + [
            "--status-file", status_file_normalized,
            "--command-file", command_file_normalized,
            "--battery-id", battery_id,
            "--pmax-kw", str(max_power_kw),
            "--max-flow-percent", str(max_flow_percent),
            "--initial-soc", str(initial_soc),
            "--interval-sec", "1",
            "--test-mode", test_mode,
        ]
        
        # 資料收集設定
        if self.collect_data_var.get():
            cmd.append("--collect-data")
            data_output_dir = self.data_output_dir_var.get()
            if data_output_dir:
                # 確保路徑使用正確的格式
                data_output_dir_normalized = os.path.normpath(data_output_dir)
                cmd.extend(["--data-output-dir", data_output_dir_normalized])
            collect_interval = self.collect_interval_var.get()
            if collect_interval:
                cmd.extend(["--collect-interval-sec", str(int(collect_interval))])
            # 電池數量設定（只收集前 N 顆電池）
            battery_count = int(self.battery_count_var.get()) if self.battery_count_var.get() else 1
            cmd.extend(["--battery-count", str(battery_count)])
            
            # 電池容量和效率設定（用於理論計算 SoC）
            # GUI 中單位是 Wh，需要轉換為 kWh 傳給 run_online_control.py
            battery_capacity_wh = float(self.battery_capacity_var.get()) if self.battery_capacity_var.get() else 10.0
            battery_capacity_kwh = battery_capacity_wh / 1000.0  # Wh → kWh
            battery_efficiency = float(self.battery_efficiency_var.get()) if self.battery_efficiency_var.get() else 0.95
            cmd.extend(["--battery-capacity-kwh", str(battery_capacity_kwh)])
            cmd.extend(["--battery-efficiency", str(battery_efficiency)])
        
        # 負載規格設定
        load_power_per_unit_w = float(self.load_power_per_unit_var.get()) if self.load_power_per_unit_var.get() else 1000.0
        max_load_count = int(self.max_load_count_var.get()) if self.max_load_count_var.get() else 10
        cmd.extend(["--load-power-per-unit-w", str(load_power_per_unit_w)])
        cmd.extend(["--max-load-count", str(max_load_count)])
        
        # 負載模式文件（如果指定）
        load_pattern_file = self.load_pattern_file_var.get()
        if load_pattern_file:
            load_pattern_file_normalized = os.path.normpath(load_pattern_file)
            cmd.extend(["--load-pattern-file", load_pattern_file_normalized])
        
        if test_mode == "ai":
            cmd.extend(["--model-path", model_path, "--use-power-to-flow", "--use-safetynet"])
        
        # 啟動程式
        self._log(f"[DEBUG] 啟動命令: {' '.join(cmd)}\n")
        self._log(f"[DEBUG] 工作目錄: {project_root}\n")
        self._log(f"[DEBUG] 狀態檔案: {status_file}\n")
        self._log(f"[DEBUG] 命令檔案: {command_file}\n")
        
        self.current_scenario = ScriptProcess(scenario_name, cmd, log_callback=self._log)
        if self.current_scenario.start(cwd=project_root):
            self.scenario_status_var.set(f"執行中: {scenario_name}")
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] [{scenario_name}] 啟動成功\n")
            
            # 如果啟用資料收集且啟用 Watchdog，啟動監控線程
            if self.collect_data_var.get() and self.use_watchdog_var.get():
                self.watchdog_cmd_backup = cmd  # 備份命令用於重啟
                self.watchdog_project_root = project_root
                self.watchdog_restart_count = 0
                self.watchdog_stop_flag.clear()
                
                check_interval = int(self.watchdog_check_interval_var.get()) if self.watchdog_check_interval_var.get() else 60
                self.watchdog_thread = threading.Thread(
                    target=self._watchdog_monitor,
                    args=(check_interval,),
                    daemon=True,
                    name="WatchdogMonitor"
                )
                self.watchdog_thread.start()
                self._log(f"[{datetime.now().strftime('%H:%M:%S')}] [Watchdog] 已啟動，檢查間隔: {check_interval} 秒\n")
            
            # 如果啟用 SoC/SoH 計算，啟動計算進程
            if self.enable_soc_soh_var.get():
                self._start_soc_soh_calculation(command_file_normalized, project_root, battery_id)
        else:
            self.scenario_status_var.set("啟動失敗")
            self._log(f"[錯誤] 啟動失敗，請檢查日誌\n")
    
    def _start_soc_soh_calculation(self, command_file: str, project_root: str, battery_id: str):
        """啟動 SoC/SoH 計算進程"""
        try:
            # 獲取參數
            initial_soc = float(self.initial_soc_var.get()) if self.initial_soc_var.get() else 50.0
            battery_capacity_wh = float(self.battery_capacity_var.get()) if self.battery_capacity_var.get() else 10.0
            battery_capacity_kwh = battery_capacity_wh / 1000.0  # Wh → kWh
            battery_efficiency = float(self.battery_efficiency_var.get()) if self.battery_efficiency_var.get() else 0.95
            read_interval_sec = float(self.soc_soh_read_interval_var.get()) if self.soc_soh_read_interval_var.get() else 1.0
            output_csv = self.soc_soh_output_csv_var.get()
            
            # 構建命令
            if getattr(sys, 'frozen', False):
                # EXE 模式
                exe_dir = os.path.dirname(sys.executable)
                script_path = os.path.join(exe_dir, "scripts", "calculate_soc_soh.py")
                if not os.path.exists(script_path):
                    # 如果找不到腳本，嘗試使用 Python 執行
                    app_dir = os.path.dirname(os.path.abspath(__file__))
                    script_path = os.path.join(os.path.dirname(app_dir), "scripts", "calculate_soc_soh.py")
                py_cmd = [sys.executable, script_path]
            else:
                # 開發模式
                app_dir = os.path.dirname(os.path.abspath(__file__))
                script_path = os.path.join(os.path.dirname(app_dir), "scripts", "calculate_soc_soh.py")
                py_cmd = [sys.executable, script_path]
            
            cmd = py_cmd + [
                "--command-file", command_file,
                "--initial-soc", str(initial_soc),
                "--battery-capacity-kwh", str(battery_capacity_kwh),
                "--battery-efficiency", str(battery_efficiency),
                "--read-interval-sec", str(read_interval_sec),
                "--battery-id", battery_id,
                "--output-csv", output_csv,
            ]
            
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] [SoC/SoH] 啟動計算進程...\n")
            self._log(f"[SoC/SoH] 命令: {' '.join(cmd)}\n")
            
            self.soc_soh_process = ScriptProcess("SoC/SoH計算", cmd, log_callback=self._log)
            if self.soc_soh_process.start(cwd=project_root):
                self._log(f"[{datetime.now().strftime('%H:%M:%S')}] [SoC/SoH] 計算進程已啟動\n")
            else:
                self._log(f"[錯誤] [SoC/SoH] 計算進程啟動失敗\n")
                self.soc_soh_process = None
        except Exception as e:
            self._log(f"[錯誤] [SoC/SoH] 啟動計算進程時發生錯誤: {e}\n")
            import traceback
            self._log(f"[錯誤] [SoC/SoH] 錯誤詳情:\n{traceback.format_exc()}\n")
            self.soc_soh_process = None
    
    def _stop_scenario(self):
        """停止當前場景"""
        # 先停止 SoC/SoH 計算
        if self.soc_soh_process:
            self.soc_soh_process.stop()
            self.soc_soh_process = None
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] [SoC/SoH] 已停止計算\n")
        
        # 先停止 Watchdog 監控
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            self.watchdog_stop_flag.set()
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] [Watchdog] 已停止監控\n")
            # 等待線程結束（最多 2 秒）
            self.watchdog_thread.join(timeout=2.0)
            self.watchdog_thread = None
        
        if self.current_scenario:
            self.current_scenario.stop()
            self.current_scenario = None
            self.scenario_status_var.set("已停止")
            
        # 清除備份命令
        self.watchdog_cmd_backup = []
        self.watchdog_restart_count = 0
    
    def _watchdog_monitor(self, check_interval: int):
        """Watchdog 監控線程（內建功能）
        
        定期檢查進程是否還在運行，如果停止則自動重啟
        """
        max_restart_count = 10  # 最大重啟次數（避免無限重啟）
        max_restart_window = 3600.0  # 時間窗口（秒）
        restart_times_list = []  # 記錄重啟時間
        
        self._log(f"[Watchdog] 監控線程已啟動，檢查間隔: {check_interval} 秒\n")
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        try:
            while not self.watchdog_stop_flag.is_set():
                # 等待檢查間隔
                if self.watchdog_stop_flag.wait(timeout=check_interval):
                    # 收到停止信號
                    break
                
                # 檢查進程狀態
                if self.current_scenario is None or not self.current_scenario.is_running():
                    # 進程已停止
                    consecutive_failures += 1
                    current_time = datetime.now()
                    
                    # 檢查是否超過最大重啟次數（時間窗口內）
                    time_window_start = time.time() - max_restart_window
                    restart_times_list = [t for t in restart_times_list if t > time_window_start]
                    
                    if len(restart_times_list) >= max_restart_count:
                        self._log(
                            f"[{current_time.strftime('%H:%M:%S')}] [Watchdog] 警告：在過去 {max_restart_window/60:.0f} 分鐘內已重啟 {len(restart_times_list)} 次，"
                            f"達到上限 {max_restart_count} 次，停止自動重啟\n"
                        )
                        break
                    
                    self._log(
                        f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] [Watchdog] 檢測到進程已停止（連續失敗 {consecutive_failures} 次），正在自動重啟...\n"
                    )
                    
                    # 嘗試重新啟動
                    if self.watchdog_cmd_backup and self.watchdog_project_root:
                        # 重新創建進程
                        scenario_name = "AI控制" if "--test-mode" not in self.watchdog_cmd_backup else "測試場景"
                        
                        # 查找 test-mode 參數來確定場景名稱
                        try:
                            test_mode_idx = self.watchdog_cmd_backup.index("--test-mode")
                            if test_mode_idx + 1 < len(self.watchdog_cmd_backup):
                                test_mode = self.watchdog_cmd_backup[test_mode_idx + 1]
                                if test_mode == "ai":
                                    scenario_name = "AI控制"
                                elif test_mode == "id0":
                                    scenario_name = "ID=0不控制測試"
                                else:
                                    scenario_name = "數據傳輸測試"
                        except (ValueError, IndexError):
                            pass
                        
                        self.current_scenario = ScriptProcess(scenario_name, self.watchdog_cmd_backup, log_callback=self._log)
                        
                        if self.current_scenario.start(cwd=self.watchdog_project_root):
                            # 重啟成功
                            consecutive_failures = 0
                            self.watchdog_restart_count += 1
                            restart_times_list.append(time.time())
                            current_time = datetime.now()
                            self.scenario_status_var.set(f"執行中: {scenario_name} (自動重啟 #{self.watchdog_restart_count})")
                            self._log(
                                f"[{current_time.strftime('%H:%M:%S')}] [Watchdog] 進程已成功重啟（總共重啟 {self.watchdog_restart_count} 次）\n"
                            )
                        else:
                            # 重啟失敗
                            consecutive_failures += 1
                            self._log(
                                f"[{current_time.strftime('%H:%M:%S')}] [Watchdog] 重啟失敗（連續失敗 {consecutive_failures} 次）\n"
                            )
                            
                            if consecutive_failures >= max_consecutive_failures:
                                self._log(
                                    f"[{current_time.strftime('%H:%M:%S')}] [Watchdog] 連續失敗 {consecutive_failures} 次，"
                                    f"停止自動重啟（請檢查錯誤原因）\n"
                                )
                                break
                    else:
                        # 沒有備份命令，無法重啟
                        self._log(
                            f"[{current_time.strftime('%H:%M:%S')}] [Watchdog] 無法重啟：缺少備份命令\n"
                        )
                        break
                else:
                    # 進程正常運行
                    consecutive_failures = 0
                    
        except Exception as e:
            current_time = datetime.now()
            self._log(
                f"[{current_time.strftime('%H:%M:%S')}] [Watchdog] 監控線程發生錯誤: {e}\n"
            )
            import traceback
            self._log(f"[Watchdog] 錯誤詳情:\n{traceback.format_exc()}\n")
        
        current_time = datetime.now()
        self._log(f"[{current_time.strftime('%H:%M:%S')}] [Watchdog] 監控線程已結束\n")
    
    def _cleanup_orphaned_processes(self):
        """清理殘留的 run_online_control.exe 進程（GUI 啟動時調用）"""
        try:
            # 檢查是否有殘留的 run_online_control.exe 進程
            # 使用 tasklist 查找進程
            try:
                out = subprocess.check_output(
                    ["tasklist", "/FI", "IMAGENAME eq run_online_control.exe"],
                    encoding="utf-8",
                    errors="ignore",
                    timeout=2,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                )
                
                # 檢查是否有 run_online_control.exe 進程
                if "run_online_control.exe" in out.lower():
                    # 找到殘留進程，嘗試終止
                    try:
                        # 使用 taskkill 終止進程
                        subprocess.run(
                            ["taskkill", "/F", "/IM", "run_online_control.exe"],
                            timeout=3,
                            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        print(f"[清理] 已終止殘留的 run_online_control.exe 進程")
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
    
    def _cleanup_all(self):
        """清理所有資源：停止所有場景和進程"""
        # 停止當前場景
        if self.current_scenario:
            try:
                self.current_scenario.stop()
                self.current_scenario = None
            except Exception:
                pass
        
        # 停止廠商程式（如果是由 GUI 啟動的）
        if hasattr(self, 'vendor_proc') and self.vendor_proc:
            try:
                if self.vendor_proc.poll() is None:  # 還在運行
                    self.vendor_proc.terminate()
                    try:
                        self.vendor_proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.vendor_proc.kill()
            except Exception:
                pass
            finally:
                self.vendor_proc = None
    
    def _on_closing(self):
        """窗口關閉時的處理"""
        # 清理所有資源
        self._cleanup_all()
        # 關閉窗口
        self.destroy()
    
    def _browse_vendor_dir(self):
        dir_path = filedialog.askdirectory(title="選擇廠商程式資料夾", initialdir=self.vendor_dir_var.get())
        if dir_path:
            self.vendor_dir_var.set(dir_path)
            exe_path = os.path.join(dir_path, "P302.exe")
            if os.path.exists(exe_path):
                self.vendor_exe_var.set(exe_path)
    
    def _browse_vendor_exe(self):
        exe_path = filedialog.askopenfilename(
            title="選擇廠商程式 EXE",
            initialdir=os.path.dirname(self.vendor_exe_var.get()) if self.vendor_exe_var.get() else "",
            filetypes=[("執行檔", "*.exe"), ("所有檔案", "*.*")]
        )
        if exe_path:
            self.vendor_exe_var.set(exe_path)
            self.vendor_dir_var.set(os.path.dirname(exe_path))
    
    def _browse_model(self):
        model_path = filedialog.askopenfilename(
            title="選擇模型檔案",
            initialdir=os.path.dirname(self.model_path_var.get()) if self.model_path_var.get() else "",
            filetypes=[("模型檔案", "*.pth"), ("所有檔案", "*.*")]
        )
        if model_path:
            self.model_path_var.set(model_path)
    
    def _browse_data_output_dir(self):
        output_dir = filedialog.askdirectory(
            title="選擇資料收集輸出目錄",
            initialdir=self.data_output_dir_var.get() if self.data_output_dir_var.get() else ""
        )
        if output_dir:
            self.data_output_dir_var.set(output_dir)
    
    def _browse_load_pattern_file(self):
        """瀏覽負載模式文件"""
        filename = filedialog.askopenfilename(
            title="選擇負載模式文件",
            filetypes=[("文字檔案", "*.txt"), ("所有檔案", "*.*")],
            initialdir=os.path.dirname(self.load_pattern_file_var.get()) if self.load_pattern_file_var.get() else ""
        )
        if filename:
            self.load_pattern_file_var.set(filename)
    
    def _browse_soc_soh_output(self):
        """瀏覽 SoC/SoH 輸出 CSV 檔案"""
        filename = filedialog.asksaveasfilename(
            title="選擇 SoC/SoH 輸出 CSV 檔案",
            defaultextension=".csv",
            filetypes=[("CSV 檔案", "*.csv"), ("所有檔案", "*.*")],
            initialdir=os.path.dirname(self.soc_soh_output_csv_var.get()) if self.soc_soh_output_csv_var.get() else ""
        )
        if filename:
            self.soc_soh_output_csv_var.set(filename)
    
    def _save_config(self):
        self.config.update({
            "vendor_dir": self.vendor_dir_var.get(),
            "vendor_exe": self.vendor_exe_var.get(),
            "model_path": self.model_path_var.get(),
            "initial_soc": float(self.initial_soc_var.get()) if self.initial_soc_var.get() else 50.0,
            "max_power_w": float(self.max_power_var.get()) if self.max_power_var.get() else 50.0,  # 改為 W
            "max_flow_percent": float(self.max_flow_var.get()) if self.max_flow_var.get() else 100.0,
            "battery_count": int(self.battery_count_var.get()) if self.battery_count_var.get() else 1,
            "battery_capacity_wh": float(self.battery_capacity_var.get()) if self.battery_capacity_var.get() else 10.0,  # 改為 Wh
            "battery_efficiency": float(self.battery_efficiency_var.get()) if self.battery_efficiency_var.get() else 0.95,
            "load_power_per_unit_w": float(self.load_power_per_unit_var.get()) if self.load_power_per_unit_var.get() else 1000.0,
            "max_load_count": int(self.max_load_count_var.get()) if self.max_load_count_var.get() else 10,
            "load_pattern_file": self.load_pattern_file_var.get(),
            "collect_data": self.collect_data_var.get(),
            "data_output_dir": self.data_output_dir_var.get(),
            "collect_interval_sec": int(self.collect_interval_var.get()) if self.collect_interval_var.get() else 900,
            "use_watchdog": self.use_watchdog_var.get(),
            "watchdog_check_interval_sec": int(self.watchdog_check_interval_var.get()) if self.watchdog_check_interval_var.get() else 60,
            "enable_soc_soh": self.enable_soc_soh_var.get(),
            "soc_soh_output_csv": self.soc_soh_output_csv_var.get(),
            "soc_soh_read_interval_sec": float(self.soc_soh_read_interval_var.get()) if self.soc_soh_read_interval_var.get() else 1.0,
        })
        
        if ConfigManager.save(self.config):
            messagebox.showinfo("成功", "設定已儲存")
        else:
            messagebox.showerror("錯誤", "設定儲存失敗")
    
    def _update_vendor_status(self):
        """更新廠商程式狀態（非阻塞）"""
        try:
            vendor_exe = self.vendor_exe_var.get()
            if not vendor_exe or not os.path.exists(vendor_exe):
                self.vendor_status_var.set("路徑無效")
                return
            
            # 使用非阻塞方式檢查（避免卡頓）
            running = check_vendor_running(vendor_exe)
            self.vendor_status_var.set("執行中" if running else "未執行")
        except Exception:
            self.vendor_status_var.set("檢查失敗")
    
    def _poll_status(self):
        """定期更新狀態"""
        if self.current_scenario:
            if self.current_scenario.is_running():
                pass  # 狀態會在日誌中更新
            else:
                self.scenario_status_var.set("已停止")
                self.current_scenario = None
        
        self.after(2000, self._poll_status)
    
    def _poll_data_command(self):
        """定期讀取並更新 Data/Command 顯示（優化：避免卡頓）"""
        try:
            vendor_dir = self.vendor_dir_var.get()
            if not vendor_dir or not os.path.exists(vendor_dir):
                # 如果路徑無效，延長輪詢間隔
                self.after(5000, self._poll_data_command)  # 5 秒後再試
                return
            
            data_file = os.path.join(vendor_dir, "Data.txt")
            cmd_file = os.path.join(vendor_dir, "Command.txt")
            
            # 讀取 Data.txt（限制檔案大小，避免讀取過大檔案）
            if os.path.exists(data_file):
                try:
                    # 檢查檔案大小，如果太大則跳過（避免卡頓）
                    file_size = os.path.getsize(data_file)
                    if file_size > 1024 * 1024:  # 超過 1MB 則跳過
                        self.after(2000, self._poll_data_command)  # 延長輪詢間隔
                        return
                    
                    with open(data_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content != self.last_data_content:
                            self.last_data_content = content
                            self._update_data_display()
                except Exception:
                    pass
            
            # 讀取 Command.txt 並更新顯示
            if os.path.exists(cmd_file):
                try:
                    file_size = os.path.getsize(cmd_file)
                    if file_size > 1024 * 1024:  # 超過 1MB 則跳過
                        self.after(2000, self._poll_data_command)
                        return
                    
                    with open(cmd_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content != self.last_command_content:
                            self.last_command_content = content
                            self._update_command_display_from_file(cmd_file)
                except Exception:
                    pass
        except Exception:
            # 發生任何錯誤時，延長輪詢間隔
            self.after(5000, self._poll_data_command)
            return
        
        self.after(1000, self._poll_data_command)  # 每秒更新一次
    
    def _open_vendor_folder(self):
        vendor_dir = self.vendor_dir_var.get()
        if os.path.exists(vendor_dir):
            try:
                os.startfile(vendor_dir)
            except Exception as e:
                messagebox.showerror("錯誤", f"無法開啟資料夾: {e}")
        else:
            messagebox.showerror("錯誤", f"資料夾不存在: {vendor_dir}")
    
    def _launch_vendor(self):
        vendor_exe = self.vendor_exe_var.get()
        if not os.path.exists(vendor_exe):
            messagebox.showerror("錯誤", f"程式不存在: {vendor_exe}")
            return
        
        try:
            # 記錄進程，以便關閉時清理
            self.vendor_proc = subprocess.Popen([vendor_exe], cwd=os.path.dirname(vendor_exe))
            self._update_vendor_status()
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] [系統] 廠商程式已啟動\n")
        except Exception as e:
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] [錯誤] 無法啟動程式: {e}\n")
            self.vendor_proc = None


def main():
    app = AIControlGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
