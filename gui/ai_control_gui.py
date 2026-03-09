#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P302 微電網 AI 控制介面 — CORAL Framework
==========================================
模式：
  1. AI 控制 (CORAL)     → control/run_deployment.py
  2. 太陽能測試 (收資料)  → control/solar_test_collect.py
  3. 待機 / 手動         → control/solar_test_collect.py (scenario 4)

P302 鋅空氣電池規格（預設）：
  容量  : 10 mAh ≈ 0.07 Wh
  充電  : 20 mA × 8.5V = 170 mW
  放電  : 20 mA × 5.6V = 112 mW
  效率  : 85% RTE
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

# ======================================================================
# 設定檔管理
# ======================================================================
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config_gui.json")

# 腳本路徑
GUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(GUI_DIR)
CONTROL_DIR = os.path.join(PROJECT_ROOT, "control")
DEPLOYMENT_SCRIPT = os.path.join(CONTROL_DIR, "run_deployment.py")
SOLAR_TEST_SCRIPT = os.path.join(CONTROL_DIR, "solar_test_collect.py")

# P302 預設值
P302_DEFAULTS = {
    "vendor_dir": "",
    "vendor_exe": "",
    "model_path": os.path.join(PROJECT_ROOT, "models", "best_sac_model.pth"),
    "initial_soc": 50.0,
    "load_count": 4,
    "log_dir": os.path.join(PROJECT_ROOT, "results", "deployment"),
    "device": "cpu",
    "poll_sec": 10.0,
    "window_min": 15,
    "use_watchdog": True,
    "watchdog_interval_sec": 60,
}


class ConfigManager:
    @staticmethod
    def load():
        defaults = dict(P302_DEFAULTS)
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    defaults.update(json.load(f))
            except Exception:
                pass
        return defaults

    @staticmethod
    def save(config):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False


# ======================================================================
# 廠商程式檢查
# ======================================================================
def check_vendor_running(vendor_exe_path: str) -> bool:
    try:
        exe_name = os.path.basename(vendor_exe_path).lower()
        out = subprocess.check_output(
            ["tasklist"], encoding="utf-8", errors="ignore", timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        return exe_name in out.lower()
    except Exception:
        return False


# ======================================================================
# 子進程管理
# ======================================================================
class ScriptProcess:
    """管理腳本進程，支援即時日誌"""

    def __init__(self, name: str, cmd: list, log_callback=None):
        self.name = name
        self.cmd = cmd
        self.proc = None
        self.log_callback = log_callback
        self._stop_logging = False

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _read_output(self, pipe):
        try:
            while not self._stop_logging:
                line = pipe.readline()
                if not line:
                    if self.proc and self.proc.poll() is not None:
                        remaining = pipe.read()
                        if remaining and self.log_callback:
                            self.log_callback(remaining if isinstance(remaining, str) else remaining.decode('utf-8', errors='replace'))
                    break
                text = line.rstrip() if isinstance(line, str) else line.decode('utf-8', errors='replace').rstrip()
                if text and self.log_callback:
                    self.log_callback(text + "\n")
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def start(self, cwd: str = None):
        if self.is_running():
            return False
        try:
            self._stop_logging = False
            self.proc = subprocess.Popen(
                self.cmd, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                bufsize=0, text=True, encoding='utf-8', errors='replace',
            )
            t = threading.Thread(target=self._read_output, args=(self.proc.stdout,), daemon=True)
            t.start()
            time.sleep(0.5)
            if self.proc.poll() is not None:
                if self.log_callback:
                    self.log_callback(f"[錯誤] 進程立即退出 (code={self.proc.returncode})\n")
                return False
            return True
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"[錯誤] 啟動失敗: {e}\n")
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


# ======================================================================
# 主 GUI
# ======================================================================
class AIControlGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("P302 微電網 AI 控制 — CORAL Framework")
        self.geometry("1400x850")

        self.config = ConfigManager.load()
        self.vendor_status_var = tk.StringVar(value="--")
        self.scenario_status_var = tk.StringVar(value="已停止")
        self.current_scenario = None
        self.vendor_proc = None

        # Watchdog
        self.watchdog_thread = None
        self.watchdog_stop_flag = threading.Event()
        self.watchdog_cmd_backup = []
        self.watchdog_project_root = ""
        self.watchdog_restart_count = 0

        # Data/Command 內容快取
        self.last_data_content = ""
        self.last_command_content = ""

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # 延遲啟動
        self.after(100, self._update_vendor_status)
        self.after(300, self._poll_status)
        self.after(500, self._poll_data_command)

    # ------------------------------------------------------------------
    # UI 建構
    # ------------------------------------------------------------------
    def _build_ui(self):
        pad = 6
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 10, "bold"))
        style.configure("Info.TLabel", font=("Consolas", 9))
        style.configure("Mode.TRadiobutton", font=("Microsoft JhengHei UI", 10))
        style.configure("Start.TButton", font=("Microsoft JhengHei UI", 10, "bold"))

        # === 主分割 ===
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill="both", expand=True, padx=pad, pady=pad)

        # ─── 左欄 ───
        left = ttk.Frame(main_paned)
        main_paned.add(left, weight=1)
        self._build_left_panel(left, pad)

        # ─── 右欄 ───
        right = ttk.Frame(main_paned)
        main_paned.add(right, weight=1)
        self._build_right_panel(right, pad)

    def _build_left_panel(self, parent, pad):
        """左欄：廠商程式設定 + 日誌 + Data/Command 顯示"""
        # ── 廠商程式 ──
        frm = ttk.LabelFrame(parent, text="廠商程式 (P302)")
        frm.pack(fill="x", padx=pad, pady=(pad, 2))

        row = ttk.Frame(frm)
        row.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row, text="程式資料夾:").pack(side="left")
        self.vendor_dir_var = tk.StringVar(value=self.config.get("vendor_dir", ""))
        ttk.Entry(row, textvariable=self.vendor_dir_var, width=35).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row, text="瀏覽", command=self._browse_vendor_dir).pack(side="left")

        row2 = ttk.Frame(frm)
        row2.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row2, text="EXE 路徑:   ").pack(side="left")
        self.vendor_exe_var = tk.StringVar(value=self.config.get("vendor_exe", ""))
        ttk.Entry(row2, textvariable=self.vendor_exe_var, width=35).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row2, text="瀏覽", command=self._browse_vendor_exe).pack(side="left")

        row_s = ttk.Frame(frm)
        row_s.pack(fill="x", padx=pad, pady=4)
        ttk.Label(row_s, text="狀態:").pack(side="left")
        ttk.Label(row_s, textvariable=self.vendor_status_var, width=10).pack(side="left", padx=4)
        ttk.Button(row_s, text="檢查", command=self._update_vendor_status).pack(side="left", padx=2)
        ttk.Button(row_s, text="啟動", command=self._launch_vendor).pack(side="left", padx=2)
        ttk.Button(row_s, text="開啟資料夾", command=self._open_vendor_folder).pack(side="left", padx=2)

        # ── 日誌 / 狀態分頁 ──
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True, padx=pad, pady=(2, pad))

        # 分頁 1：日誌
        frm_log = ttk.Frame(nb)
        nb.add(frm_log, text="執行日誌")
        self.log_text = scrolledtext.ScrolledText(frm_log, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=2, pady=2)
        self.log_text.config(state=tk.DISABLED)

        # 分頁 2：Data/Command
        frm_dc = ttk.Frame(nb)
        nb.add(frm_dc, text="Data / Command")
        paned = ttk.PanedWindow(frm_dc, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True, padx=2, pady=2)

        frm_d = ttk.LabelFrame(paned, text="Data.txt (韌體輸入)")
        paned.add(frm_d, weight=1)
        self.data_text = scrolledtext.ScrolledText(frm_d, wrap=tk.NONE, font=("Consolas", 9), height=8)
        self.data_text.pack(fill="both", expand=True, padx=2, pady=2)
        self.data_text.config(state=tk.DISABLED)

        frm_c = ttk.LabelFrame(paned, text="Command.txt (AI 輸出)")
        paned.add(frm_c, weight=1)
        self.cmd_text = scrolledtext.ScrolledText(frm_c, wrap=tk.NONE, font=("Consolas", 9), height=8)
        self.cmd_text.pack(fill="both", expand=True, padx=2, pady=2)
        self.cmd_text.config(state=tk.DISABLED)

    def _build_right_panel(self, parent, pad):
        """右欄：模式選擇 + 參數設定 + 控制按鈕"""
        # 使用 Canvas 實現滾動
        canvas = tk.Canvas(parent, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        cw = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # ── 模式選擇 ──
        frm_mode = ttk.LabelFrame(scrollable, text="運行模式")
        frm_mode.pack(fill="x", padx=pad, pady=pad)

        self.mode_var = tk.StringVar(value="ai")
        modes = [
            ("ai",    "1. AI 控制 (CORAL Framework)",
             "使用訓練好的 SAC 模型 + 安全框架\n"
             "15 分鐘聚合資料 → 推論 → 輸出 Command.txt\n"
             "包含：SoCTracker、DataBuffer、Scenario 1-4、TOU 電價"),
            ("solar", "2. 太陽能測試 (收 MPPT 資料)",
             "電池不動作 (Scenario 4)，持續收集 MPPT 資料\n"
             "指定負載組數，市電自動補足\n"
             "資料存入 CSV，可用於重新訓練"),
            ("standby", "3. 待機 (僅維持通訊)",
             "電池不動作 (Scenario 4)，負載可自訂\n"
             "每秒更新 Command.txt 維持與韌體同步"),
        ]

        for val, label, desc in modes:
            row = ttk.Frame(frm_mode)
            row.pack(fill="x", padx=pad, pady=2)
            rb = ttk.Radiobutton(row, text=label, variable=self.mode_var, value=val,
                                 style="Mode.TRadiobutton", command=self._on_mode_change)
            rb.pack(anchor="w")
            ttk.Label(row, text=desc, font=("Microsoft JhengHei UI", 8),
                      foreground="gray", justify="left").pack(anchor="w", padx=24)

        # ── AI 控制參數 ──
        self.frm_ai = ttk.LabelFrame(scrollable, text="AI 控制參數")
        self.frm_ai.pack(fill="x", padx=pad, pady=2)

        # 模型路徑
        row_m = ttk.Frame(self.frm_ai)
        row_m.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_m, text="模型檔案 (.pth):").pack(side="left")
        self.model_path_var = tk.StringVar(value=self.config.get("model_path", ""))
        ttk.Entry(row_m, textvariable=self.model_path_var, width=30).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row_m, text="瀏覽", command=self._browse_model).pack(side="left")

        # 初始 SoC
        row_soc = ttk.Frame(self.frm_ai)
        row_soc.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_soc, text="初始 SoC (%):").pack(side="left")
        self.initial_soc_var = tk.StringVar(value=str(self.config.get("initial_soc", 50.0)))
        ttk.Entry(row_soc, textvariable=self.initial_soc_var, width=10).pack(side="left", padx=4)
        ttk.Label(row_soc, text="(SoCTracker 起始值，0-100)").pack(side="left")

        # 推論裝置
        row_dev = ttk.Frame(self.frm_ai)
        row_dev.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_dev, text="推論裝置:").pack(side="left")
        self.device_var = tk.StringVar(value=self.config.get("device", "cpu"))
        ttk.Combobox(row_dev, textvariable=self.device_var, values=["cpu", "cuda"],
                     width=8, state="readonly").pack(side="left", padx=4)

        # 聚合窗格
        row_win = ttk.Frame(self.frm_ai)
        row_win.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_win, text="聚合窗格 (min):").pack(side="left")
        self.window_min_var = tk.StringVar(value=str(self.config.get("window_min", 15)))
        ttk.Entry(row_win, textvariable=self.window_min_var, width=8).pack(side="left", padx=4)
        ttk.Label(row_win, text="(預設 15 分鐘，每窗格推論一次)").pack(side="left")

        # 初始動作
        row_init = ttk.Frame(self.frm_ai)
        row_init.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_init, text="首次動作:").pack(side="left")
        self.initial_action_var = tk.StringVar(value="standby")
        ttk.Combobox(row_init, textvariable=self.initial_action_var,
                     values=["standby", "random"], width=10, state="readonly").pack(side="left", padx=4)
        ttk.Label(row_init, text="(首個 15 分鐘的動作)").pack(side="left")

        # ── 通用參數 ──
        frm_common = ttk.LabelFrame(scrollable, text="通用設定")
        frm_common.pack(fill="x", padx=pad, pady=2)

        # 負載組數
        row_load = ttk.Frame(frm_common)
        row_load.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_load, text="負載組數 (0-4):").pack(side="left")
        self.load_count_var = tk.StringVar(value=str(self.config.get("load_count", 4)))
        ttk.Spinbox(row_load, from_=0, to=4, textvariable=self.load_count_var,
                     width=5).pack(side="left", padx=4)
        ttk.Label(row_load, text="(每組 8W，4 組 = 32W)").pack(side="left")

        # 電池 PP ID
        row_pp = ttk.Frame(frm_common)
        row_pp.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_pp, text="電池 PP ID:").pack(side="left")
        self.battery_pp_var = tk.StringVar(value="01")
        ttk.Entry(row_pp, textvariable=self.battery_pp_var, width=5).pack(side="left", padx=4)
        ttk.Label(row_pp, text="(01-10，預設 01)").pack(side="left")

        # 輪詢間隔
        row_poll = ttk.Frame(frm_common)
        row_poll.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_poll, text="Data.txt 輪詢 (秒):").pack(side="left")
        self.poll_sec_var = tk.StringVar(value=str(self.config.get("poll_sec", 10.0)))
        ttk.Entry(row_poll, textvariable=self.poll_sec_var, width=8).pack(side="left", padx=4)
        ttk.Label(row_poll, text="(所有模式共用，預設 10 秒)").pack(side="left")

        # 日誌目錄
        row_log = ttk.Frame(frm_common)
        row_log.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_log, text="日誌目錄:").pack(side="left")
        self.log_dir_var = tk.StringVar(value=self.config.get("log_dir", ""))
        ttk.Entry(row_log, textvariable=self.log_dir_var, width=25).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row_log, text="瀏覽", command=self._browse_log_dir).pack(side="left")

        # ── Watchdog ──
        frm_wd = ttk.LabelFrame(scrollable, text="Watchdog (自動重啟)")
        frm_wd.pack(fill="x", padx=pad, pady=2)

        self.use_watchdog_var = tk.BooleanVar(value=self.config.get("use_watchdog", True))
        ttk.Checkbutton(frm_wd, text="啟用 Watchdog (進程異常停止時自動重啟)",
                        variable=self.use_watchdog_var).pack(anchor="w", padx=pad, pady=2)

        row_wd = ttk.Frame(frm_wd)
        row_wd.pack(fill="x", padx=pad, pady=2)
        ttk.Label(row_wd, text="檢查間隔 (秒):").pack(side="left")
        self.wd_interval_var = tk.StringVar(value=str(self.config.get("watchdog_interval_sec", 60)))
        ttk.Entry(row_wd, textvariable=self.wd_interval_var, width=8).pack(side="left", padx=4)

        # ── P302 電池規格 (唯讀資訊) ──
        frm_info = ttk.LabelFrame(scrollable, text="P302 SLFB 電池規格 (參考)")
        frm_info.pack(fill="x", padx=pad, pady=2)

        info_text = (
            "  容量: 11833.6 mAh = 66.28 Wh (模擬 4 組模組並聯)\n"
            "  功率: 170 mW (P302 硬體極限: 20mA × 8.5V)\n"
            "  充電: 8.5V  放電: 5.6V  效率: 85% RTE\n"
            "  Command.txt: 功率(mW) + 流速(0-100%) + Scenario(1-4)\n"
            "  Scenario: 1=放電全包 2=放電+市電 3=市電充電 4=待機"
        )
        ttk.Label(frm_info, text=info_text, font=("Consolas", 8),
                  justify="left").pack(anchor="w", padx=pad, pady=4)

        # ── 控制按鈕 ──
        frm_ctrl = ttk.Frame(scrollable)
        frm_ctrl.pack(fill="x", padx=pad, pady=pad)

        ttk.Label(frm_ctrl, text="狀態:").pack(side="left")
        ttk.Label(frm_ctrl, textvariable=self.scenario_status_var, width=25).pack(side="left", padx=4)

        btn_frame = ttk.Frame(scrollable)
        btn_frame.pack(fill="x", padx=pad, pady=2)

        self.btn_start = ttk.Button(btn_frame, text="▶ 啟動", command=self._start_scenario,
                                    style="Start.TButton")
        self.btn_start.pack(side="left", padx=4)
        ttk.Button(btn_frame, text="■ 停止", command=self._stop_scenario).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="儲存設定", command=self._save_config).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="清除日誌", command=self._clear_log).pack(side="left", padx=4)

        # 初始化模式顯示
        self._on_mode_change()

    # ------------------------------------------------------------------
    # 模式切換
    # ------------------------------------------------------------------
    def _on_mode_change(self):
        """根據選擇的模式顯示/隱藏 AI 參數"""
        mode = self.mode_var.get()
        if mode == "ai":
            self.frm_ai.pack(fill="x", padx=6, pady=2, after=self.frm_ai.master.winfo_children()[0])
            # 確保 frm_ai 在 mode frame 之後
            for w in self.frm_ai.master.winfo_children():
                if isinstance(w, ttk.LabelFrame) and w.cget("text") == "運行模式":
                    self.frm_ai.pack(fill="x", padx=6, pady=2, after=w)
                    break
        else:
            # AI 參數不需要顯示，但保留在 layout 中
            pass

    # ------------------------------------------------------------------
    # 日誌
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg)
        # 限制日誌長度
        if float(self.log_text.index('end-1c').split('.')[0]) > 5000:
            self.log_text.delete('1.0', '1000.0')
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        for widget in [self.log_text, self.data_text, self.cmd_text]:
            widget.config(state=tk.NORMAL)
            widget.delete(1.0, tk.END)
            widget.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # 啟動場景
    # ------------------------------------------------------------------
    def _start_scenario(self):
        if self.current_scenario and self.current_scenario.is_running():
            messagebox.showwarning("警告", "已有場景在執行中，請先停止")
            return

        mode = self.mode_var.get()
        vendor_dir = self.vendor_dir_var.get()

        if not vendor_dir or not os.path.isdir(vendor_dir):
            messagebox.showerror("錯誤", "請先設定正確的廠商程式資料夾")
            return

        data_file = os.path.join(vendor_dir, "Data.txt")
        command_file = os.path.join(vendor_dir, "Command.txt")
        pp = self.battery_pp_var.get().strip() or "01"
        load_count = int(self.load_count_var.get() or 4)

        if mode == "ai":
            cmd, scenario_name = self._build_ai_cmd(data_file, command_file, pp)
        elif mode == "solar":
            cmd, scenario_name = self._build_solar_cmd(data_file, command_file, pp, load_count)
        else:  # standby
            cmd, scenario_name = self._build_standby_cmd(data_file, command_file, pp, load_count)

        if cmd is None:
            return

        self._log(f"\n{'='*60}\n")
        self._log(f"[{datetime.now().strftime('%H:%M:%S')}] 啟動: {scenario_name}\n")
        self._log(f"  命令: {' '.join(cmd)}\n")
        self._log(f"  工作目錄: {PROJECT_ROOT}\n")
        self._log(f"{'='*60}\n\n")

        self.current_scenario = ScriptProcess(scenario_name, cmd, log_callback=self._log)
        if self.current_scenario.start(cwd=PROJECT_ROOT):
            self.scenario_status_var.set(f"執行中: {scenario_name}")
            self._log(f"[OK] 進程已啟動 (PID: {self.current_scenario.proc.pid})\n")

            # Watchdog
            if self.use_watchdog_var.get():
                self._start_watchdog(cmd, scenario_name)
        else:
            self.scenario_status_var.set("啟動失敗")

    def _build_ai_cmd(self, data_file, command_file, pp):
        """建構 AI 控制命令"""
        model_path = self.model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("錯誤", f"模型檔案不存在: {model_path}")
            return None, None

        try:
            initial_soc_pct = float(self.initial_soc_var.get())
            initial_soc = initial_soc_pct / 100.0  # % → 0~1
        except ValueError:
            messagebox.showerror("錯誤", "初始 SoC 格式錯誤")
            return None, None

        device = self.device_var.get()
        window_min = self.window_min_var.get() or "15"
        poll_sec = self.poll_sec_var.get() or "10"
        initial_action = self.initial_action_var.get()
        log_dir = self.log_dir_var.get() or os.path.join(PROJECT_ROOT, "results", "deployment")

        # 確定腳本路徑
        if getattr(sys, 'frozen', False):
            # PyInstaller EXE：用 --mode deployment 讓自己切換到部署模式
            cmd = [sys.executable, "--mode", "deployment"]
        else:
            cmd = [sys.executable, DEPLOYMENT_SCRIPT]

        cmd += [
            "--data-file", os.path.normpath(data_file),
            "--command-file", os.path.normpath(command_file),
            "--model-path", os.path.normpath(model_path),
            "--battery-pp", pp,
            "--initial-soc", str(initial_soc),
            "--poll-sec", str(poll_sec),
            "--window-min", str(window_min),
            "--device", device,
            "--log-dir", os.path.normpath(log_dir),
            "--initial-action", initial_action,
        ]

        return cmd, "AI 控制 (CORAL)"

    def _build_solar_cmd(self, data_file, command_file, pp, load_count):
        """建構太陽能測試命令"""
        log_dir = self.log_dir_var.get() or os.path.join(PROJECT_ROOT, "results", "solar_test")
        poll_sec = self.poll_sec_var.get() or "10"

        if getattr(sys, 'frozen', False):
            # PyInstaller EXE：用 --mode solar_test 讓自己切換到太陽能測試模式
            cmd = [sys.executable, "--mode", "solar_test"]
        else:
            cmd = [sys.executable, SOLAR_TEST_SCRIPT]

        cmd += [
            "--data-file", os.path.normpath(data_file),
            "--command-file", os.path.normpath(command_file),
            "--battery-pp", pp,
            "--load-count", str(load_count),
            "--scenario", "4",
            "--poll-sec", str(poll_sec),
            "--log-dir", os.path.normpath(log_dir),
        ]

        return cmd, "太陽能測試"

    def _build_standby_cmd(self, data_file, command_file, pp, load_count):
        """建構待機命令（重用 solar_test_collect.py）"""
        log_dir = self.log_dir_var.get() or os.path.join(PROJECT_ROOT, "results", "standby")
        poll_sec = self.poll_sec_var.get() or "10"

        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, "--mode", "solar_test"]
        else:
            cmd = [sys.executable, SOLAR_TEST_SCRIPT]

        cmd += [
            "--data-file", os.path.normpath(data_file),
            "--command-file", os.path.normpath(command_file),
            "--battery-pp", pp,
            "--load-count", str(load_count),
            "--scenario", "4",
            "--poll-sec", str(poll_sec),
            "--log-dir", os.path.normpath(log_dir),
        ]

        return cmd, "待機"

    # ------------------------------------------------------------------
    # 停止場景
    # ------------------------------------------------------------------
    def _stop_scenario(self):
        # 停止 Watchdog
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            self.watchdog_stop_flag.set()
            self.watchdog_thread.join(timeout=2)
            self.watchdog_thread = None
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] [Watchdog] 已停止\n")

        if self.current_scenario:
            self.current_scenario.stop()
            self.current_scenario = None
            self.scenario_status_var.set("已停止")

        self.watchdog_cmd_backup = []
        self.watchdog_restart_count = 0

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------
    def _start_watchdog(self, cmd, scenario_name):
        self.watchdog_cmd_backup = cmd
        self.watchdog_project_root = PROJECT_ROOT
        self.watchdog_restart_count = 0
        self.watchdog_stop_flag.clear()

        interval = int(self.wd_interval_var.get() or 60)
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop, args=(interval, scenario_name), daemon=True)
        self.watchdog_thread.start()
        self._log(f"[Watchdog] 已啟動 (間隔 {interval}s)\n")

    def _watchdog_loop(self, interval, scenario_name):
        max_restarts = 10
        restart_times = []

        try:
            while not self.watchdog_stop_flag.is_set():
                if self.watchdog_stop_flag.wait(timeout=interval):
                    break

                if self.current_scenario is None or not self.current_scenario.is_running():
                    # 檢查重啟次數
                    now = time.time()
                    restart_times = [t for t in restart_times if t > now - 3600]
                    if len(restart_times) >= max_restarts:
                        self._log(f"[Watchdog] 重啟過於頻繁，停止自動重啟\n")
                        break

                    self._log(f"[{datetime.now().strftime('%H:%M:%S')}] [Watchdog] 偵測到停止，自動重啟...\n")
                    self.current_scenario = ScriptProcess(scenario_name, self.watchdog_cmd_backup,
                                                          log_callback=self._log)
                    if self.current_scenario.start(cwd=self.watchdog_project_root):
                        self.watchdog_restart_count += 1
                        restart_times.append(now)
                        self.scenario_status_var.set(f"執行中: {scenario_name} (重啟 #{self.watchdog_restart_count})")
                        self._log(f"[Watchdog] 重啟成功 (#{self.watchdog_restart_count})\n")
                    else:
                        self._log(f"[Watchdog] 重啟失敗\n")
        except Exception as e:
            self._log(f"[Watchdog] 錯誤: {e}\n")

    # ------------------------------------------------------------------
    # 狀態輪詢
    # ------------------------------------------------------------------
    def _poll_status(self):
        if self.current_scenario:
            if not self.current_scenario.is_running():
                self.scenario_status_var.set("已停止 (進程結束)")
                self.current_scenario = None
        self.after(2000, self._poll_status)

    def _poll_data_command(self):
        """定期讀取並更新 Data.txt / Command.txt 顯示"""
        vendor_dir = self.vendor_dir_var.get()
        if not vendor_dir or not os.path.isdir(vendor_dir):
            self.after(5000, self._poll_data_command)
            return

        # Data.txt
        data_file = os.path.join(vendor_dir, "Data.txt")
        if os.path.exists(data_file):
            try:
                size = os.path.getsize(data_file)
                if size < 10240:  # < 10KB
                    with open(data_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content != self.last_data_content:
                        self.last_data_content = content
                        self._update_text_widget(self.data_text, self._format_data_display(content))
            except Exception:
                pass

        # Command.txt
        cmd_file = os.path.join(vendor_dir, "Command.txt")
        if os.path.exists(cmd_file):
            try:
                size = os.path.getsize(cmd_file)
                if size < 10240:
                    with open(cmd_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content != self.last_command_content:
                        self.last_command_content = content
                        self._update_text_widget(self.cmd_text, self._format_command_display(content))
            except Exception:
                pass

        self.after(1000, self._poll_data_command)

    def _format_data_display(self, raw: str) -> str:
        """格式化 Data.txt 顯示（支援新版 MPPT-Bus + 負載格式）"""
        if not raw:
            return "(空)"
        lines = [ln.strip() for ln in raw.strip().split('\n') if ln.strip()]
        out = []
        idx = 0
        
        # Line 1: 時間戳
        if idx < len(lines):
            parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
            if len(parts[0]) >= 14 and parts[0][:14].isdigit():
                ts = parts[0][:14]
                load_info = f", 負載={parts[1]}組" if len(parts) > 1 else ""
                out.append(f"時間: {ts[:4]}/{ts[4:6]}/{ts[6:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]}{load_info}")
                idx += 1
        
        # Line 2: MPPT 行（6 或 9 欄位）
        has_bus = False
        if idx < len(lines):
            parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
            if len(parts) >= 6:
                first_field = parts[0]
                is_battery = first_field.isdigit() and 1 <= int(first_field) <= 10
                if not is_battery:
                    try:
                        sv = float(parts[0]) / 100.0
                        si = float(parts[1])
                        sp = float(parts[2])
                        mv = float(parts[3]) / 100.0
                        mi = float(parts[4])
                        mp = float(parts[5])
                        out.append(f"Solar: {sv:.2f}V {si:.0f}mA {sp:.0f}mW ({sp/1000:.3f}W)")
                        out.append(f"MPPT : {mv:.2f}V {mi:.0f}mA {mp:.0f}mW ({mp/1000:.3f}W)")
                        # 新格式：MPPT-Bus（≥9 欄位）
                        if len(parts) >= 9:
                            bv2 = float(parts[6]) / 100.0
                            bi2 = float(parts[7])
                            bp2 = float(parts[8])
                            out.append(f"Bus  : {bv2:.2f}V {bi2:.0f}mA {bp2:.0f}mW ({bp2/1000:.3f}W)")
                            has_bus = True
                    except (ValueError, IndexError):
                        out.append(f"MPPT: {lines[idx]}")
                    idx += 1
        
        # Line 3 (新格式): 負載行（3 欄位，在 MPPT-Bus 之後）
        if idx < len(lines) and has_bus:
            parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
            if len(parts) >= 3:
                first_field = parts[0]
                is_battery = (first_field.isdigit() and 1 <= int(first_field) <= 10
                              and len(parts) >= 6)
                if not is_battery:
                    try:
                        lv = float(parts[0]) / 100.0
                        li = float(parts[1])
                        lp = float(parts[2])
                        out.append(f"負載 : {lv:.2f}V {li:.0f}mA {lp:.0f}mW ({lp/1000:.3f}W)")
                    except (ValueError, IndexError):
                        out.append(f"Load: {lines[idx]}")
                    idx += 1
        
        # 剩餘行: 電池資料
        while idx < len(lines):
            line = lines[idx]
            idx += 1
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) >= 6:
                try:
                    pp = parts[0]
                    soc = float(parts[1]) / 10.0
                    bv = float(parts[2]) / 100.0
                    bi = float(parts[3])
                    temp = float(parts[4]) / 10.0
                    speed = float(parts[5]) / 10.0
                    out.append(f"電池{pp}: SoC={soc:.1f}% V={bv:.2f}V I={bi:.0f}mA T={temp:.1f}°C 流速={speed:.0f}%")
                except (ValueError, IndexError):
                    out.append(line)
            else:
                    out.append(line)
        return '\n'.join(out) if out else raw

    def _format_command_display(self, raw: str) -> str:
        """格式化 Command.txt 顯示"""
        if not raw:
            return "(空)"
        lines = raw.strip().split('\n')
        out = []
        scenario_names = {1: "放電全包", 2: "放電+市電", 3: "市電充電", 4: "待機"}

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 情況碼行 (1~4)
            if len(line) == 1 and line in "1234":
                code = int(line)
                out.append(f"情況碼: {code} ({scenario_names.get(code, '?')})")
            # 時間戳行
            elif len(line) >= 14 and line[:14].isdigit():
                ts = line[:14]
                rest = line[14:]
                load_info = ""
                if rest.startswith(','):
                    parts = rest.split(',')
                    if len(parts) >= 2 and parts[1].strip().isdigit():
                        load_info = f", 負載={parts[1].strip()}組"
                out.append(f"時間: {ts[:4]}/{ts[4:6]}/{ts[6:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]}{load_info}")
            # 電池命令行 (PP,power_mW,flow_pct,)
            else:
                parts = [p.strip() for p in line.split(',') if p.strip()]
                if len(parts) >= 3:
                    try:
                        pp = parts[0]
                        power_mw = int(parts[1])
                        flow = int(parts[2])
                        power_w = power_mw / 1000.0
                        out.append(f"  電池{pp}: {power_mw}mW ({power_w:.3f}W) 流速={flow}%")
                    except (ValueError, IndexError):
                        out.append(f"  {line}")
                else:
                    out.append(f"  {line}")
        return '\n'.join(out) if out else raw

    def _update_text_widget(self, widget, text):
        widget.config(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # 瀏覽按鈕
    # ------------------------------------------------------------------
    def _browse_vendor_dir(self):
        d = filedialog.askdirectory(title="選擇廠商程式資料夾",
                                     initialdir=self.vendor_dir_var.get() or "")
        if d:
            self.vendor_dir_var.set(d)
            exe = os.path.join(d, "P302.exe")
            if os.path.exists(exe):
                self.vendor_exe_var.set(exe)

    def _browse_vendor_exe(self):
        f = filedialog.askopenfilename(
            title="選擇廠商程式 EXE",
            filetypes=[("執行檔", "*.exe"), ("所有", "*.*")])
        if f:
            self.vendor_exe_var.set(f)
            self.vendor_dir_var.set(os.path.dirname(f))

    def _browse_model(self):
        f = filedialog.askopenfilename(
            title="選擇 SAC 模型",
            initialdir=os.path.dirname(self.model_path_var.get()) if self.model_path_var.get() else "",
            filetypes=[("模型", "*.pth"), ("所有", "*.*")])
        if f:
            self.model_path_var.set(f)

    def _browse_log_dir(self):
        d = filedialog.askdirectory(title="選擇日誌輸出目錄",
                                     initialdir=self.log_dir_var.get() or "")
        if d:
            self.log_dir_var.set(d)

    # ------------------------------------------------------------------
    # 廠商程式
    # ------------------------------------------------------------------
    def _update_vendor_status(self):
        exe = self.vendor_exe_var.get()
        if not exe or not os.path.exists(exe):
            self.vendor_status_var.set("路徑無效")
            return
        running = check_vendor_running(exe)
        self.vendor_status_var.set("執行中 ✓" if running else "未執行")

    def _launch_vendor(self):
        exe = self.vendor_exe_var.get()
        if not os.path.exists(exe):
            messagebox.showerror("錯誤", f"找不到: {exe}")
            return
        try:
            self.vendor_proc = subprocess.Popen([exe], cwd=os.path.dirname(exe))
            self._update_vendor_status()
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] 廠商程式已啟動\n")
        except Exception as e:
            self._log(f"[錯誤] 無法啟動: {e}\n")

    def _open_vendor_folder(self):
        d = self.vendor_dir_var.get()
        if os.path.isdir(d):
            os.startfile(d)
        else:
            messagebox.showerror("錯誤", f"資料夾不存在: {d}")

    # ------------------------------------------------------------------
    # 設定
    # ------------------------------------------------------------------
    def _save_config(self):
        self.config.update({
            "vendor_dir": self.vendor_dir_var.get(),
            "vendor_exe": self.vendor_exe_var.get(),
            "model_path": self.model_path_var.get(),
            "initial_soc": float(self.initial_soc_var.get() or 50),
            "load_count": int(self.load_count_var.get() or 4),
            "log_dir": self.log_dir_var.get(),
            "device": self.device_var.get(),
            "poll_sec": float(self.poll_sec_var.get() or 10),
            "window_min": int(self.window_min_var.get() or 15),
            "use_watchdog": self.use_watchdog_var.get(),
            "watchdog_interval_sec": int(self.wd_interval_var.get() or 60),
        })
        if ConfigManager.save(self.config):
            messagebox.showinfo("成功", "設定已儲存")
        else:
            messagebox.showerror("錯誤", "儲存失敗")

    # ------------------------------------------------------------------
    # 關閉
    # ------------------------------------------------------------------
    def _on_closing(self):
        self._stop_scenario()
        if self.vendor_proc:
            try:
                if self.vendor_proc.poll() is None:
                    self.vendor_proc.terminate()
            except Exception:
                pass
        self.destroy()


# ======================================================================
# Entry point
# ======================================================================
def _resolve_control_path():
    """找到 control/ 目錄並加入 sys.path，確保子腳本可匯入"""
    candidates = [
        CONTROL_DIR,                                         # 原始碼模式
        os.path.join(os.path.dirname(sys.executable), "_internal", "control"),  # PyInstaller one-dir
    ]
    if hasattr(sys, '_MEIPASS'):
        candidates.insert(0, os.path.join(sys._MEIPASS, "control"))

    for d in candidates:
        if os.path.isdir(d):
            if d not in sys.path:
                sys.path.insert(0, d)
            parent = os.path.dirname(d)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return d
    return None


def main():
    """
    主入口。支援 --mode 參數以在 PyInstaller EXE 中切換模式：
      P302_AI_GUI.exe                          → GUI
      P302_AI_GUI.exe --mode solar_test [...]  → solar_test_collect.main()
      P302_AI_GUI.exe --mode deployment [...]  → run_deployment.main()
    """
    # 檢查是否為子模式（由 GUI 的 subprocess 呼叫）
    if len(sys.argv) >= 3 and sys.argv[1] == "--mode":
        mode = sys.argv[2]
        # 移除 --mode <mode>，讓子腳本的 argparse 正常工作
        sys.argv = [sys.argv[0]] + sys.argv[3:]

        _resolve_control_path()

        if mode == "solar_test":
            from solar_test_collect import main as solar_main
            solar_main()
            return
        elif mode == "deployment":
            from run_deployment import main as deploy_main
            deploy_main()
            return
        else:
            print(f"Unknown mode: {mode}", file=sys.stderr)
            sys.exit(1)

    # 預設：啟動 GUI
    app = AIControlGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
