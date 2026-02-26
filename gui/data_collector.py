#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
資料收集模組：記錄 Data.txt 內容到 CSV 檔案，用於後續模型訓練
"""

import os
import csv
from datetime import datetime
from typing import Dict, Optional, Tuple
try:
    from io_protocol import TZ_UTC8
except ImportError:
    # PyInstaller EXE 可能將模組打成 app.io_protocol
    from app.io_protocol import TZ_UTC8


class DataCollector:
    """資料收集器，將 Data.txt 內容記錄到 CSV 檔案，並理論計算 SoC 變化"""
    
    def __init__(self, output_dir: str, collect_interval_sec: int = 900, 
                 battery_capacity_kwh: float = 10.0, 
                 battery_efficiency: float = 0.95,
                 command_file_path: Optional[str] = None,
                 load_power_per_unit_w: float = 1000.0):
        """
        初始化資料收集器
        
        Args:
            output_dir: 輸出目錄
            collect_interval_sec: 收集間隔（秒），預設 900 秒（15 分鐘）
            battery_capacity_kwh: 電池容量（kWh），用於理論計算 SoC 變化
            battery_efficiency: 電池效率（0-1），預設 0.95（95%）
            command_file_path: Command.txt 檔案路徑，用於讀取功率命令來計算 SoC
            load_power_per_unit_w: 每顆負載功率（W），用於計算負載總功率
        """
        self.output_dir = output_dir
        self.collect_interval_sec = collect_interval_sec
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_efficiency = battery_efficiency
        self.command_file_path = command_file_path
        self.load_power_per_unit_w = load_power_per_unit_w
        
        # 驗證電池容量是否合理（發出警告但不阻止）
        if battery_capacity_kwh < 0.01:  # 小於 10 Wh（0.01 kWh）
            print(f"[警告] 電池容量非常小 ({battery_capacity_kwh} kWh)，可能導致 SoC 變化過大")
        elif battery_capacity_kwh < 0.1:  # 小於 100 Wh（0.1 kWh）
            print(f"[提示] 電池容量較小 ({battery_capacity_kwh} kWh)，請確認功率設定是否合理")
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV 檔案路徑（會根據日期動態更新）
        self.csv_file: Optional[str] = None
        
        # 記錄上次收集時間
        self.last_collect_time: Optional[datetime] = None
        self.current_date: Optional[str] = None  # 當前檔案對應的日期（YYYY-MM-DD）
        
        # 追蹤每個電池的上次狀態：{battery_id: (last_soc_pct, last_time, last_power_w)}
        self.battery_state: Dict[str, Tuple[float, datetime, float]] = {}
    
    def _get_csv_file_for_date(self, date_str: str) -> str:
        """
        根據日期取得對應的 CSV 檔案路徑
        
        Args:
            date_str: 日期字串（格式：YYYY-MM-DD）
            
        Returns:
            CSV 檔案完整路徑
        """
        filename = f"collected_data_{date_str}.csv"
        return os.path.join(self.output_dir, filename)
    
    def _ensure_csv_file_for_date(self, date_str: str):
        """
        確保指定日期的 CSV 檔案存在，如果不存在則建立並寫入標題行
        
        Args:
            date_str: 日期字串（格式：YYYY-MM-DD）
        """
        csv_file = self._get_csv_file_for_date(date_str)
        
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                # 標題行
                writer.writerow([
                    "timestamp",           # 時間戳
                    "battery_id",          # 電池 ID
                    "soc_percent",         # SoC (%)
                    "voltage_v",           # 電壓 (V)
                    "current_ma",          # 電流 (mA)
                    "temp_c",              # 溫度 (°C)
                    "speed_percent",       # 流速 (%)
                    "solar_v",             # 太陽能電壓 (V)
                    "solar_i_ma",          # 太陽能電流 (mA)
                    "solar_p_mw",          # 太陽能功率 (mW)
                    "mppt_v",              # MPPT 電壓 (V)
                    "mppt_i_ma",           # MPPT 電流 (mA)
                    "mppt_p_mw",           # MPPT 功率 (mW)
                    "load_count",          # 負載顆數（廠商回報）
                    "load_power_w",        # 負載總功率 (W) = 顆數 × 每顆功率
                ])
        
        self.csv_file = csv_file
        self.current_date = date_str  # 更新當前日期
        self.current_date = date_str
    
    def _read_command_power(self, battery_id: str) -> Optional[float]:
        """
        從 Command.txt 讀取指定電池的功率命令（W）
        
        Args:
            battery_id: 電池 ID（例如 "01", "1"）
            
        Returns:
            功率（W），如果讀取失敗或找不到則返回 None
        """
        if not self.command_file_path or not os.path.exists(self.command_file_path):
            return None
        
        try:
            with open(self.command_file_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
            
            # 第一行是時間戳，跳過
            if not lines:
                return None
            
            # 從第二行開始查找對應的電池 ID
            target_id = battery_id.lstrip('0') if battery_id.startswith('0') else battery_id
            target_id_2digit = f"{int(target_id):02d}" if target_id.isdigit() else battery_id
            
            for line in lines[1:]:  # 跳過第一行時間戳
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) >= 2:
                    cmd_battery_id = parts[0].strip()
                    power_str = parts[1].strip()
                    
                    # 比對電池 ID（支援 "01" 和 "1" 兩種格式）
                    if (cmd_battery_id == battery_id or 
                        cmd_battery_id == target_id or 
                        cmd_battery_id == target_id_2digit):
                        try:
                            # Command.txt 中的功率是 mW，轉換為 W
                            power_mw = int(power_str)
                            power_w = power_mw / 1000.0
                            return power_w
                        except (ValueError, IndexError):
                            continue
        except Exception:
            pass
        
        return None
    
    def _calculate_soc_change(self, power_w: float, time_interval_hours: float) -> float:
        """
        理論計算 SoC 變化量（%）
        
        Args:
            power_w: 功率（W），正值為充電，負值為放電
            time_interval_hours: 時間間隔（小時）
            
        Returns:
            SoC 變化量（%）
        """
        if self.battery_capacity_kwh <= 0:
            return 0.0
        
        # 轉換功率為 kW
        power_kw = power_w / 1000.0
        
        # 計算能量變化（kWh）
        energy_change_kwh = power_kw * time_interval_hours
        
        # 應用效率
        if energy_change_kwh > 0:  # 充電
            energy_change_kwh *= self.battery_efficiency
        else:  # 放電
            energy_change_kwh /= self.battery_efficiency
        
        # 計算 SoC 變化（%）
        soc_change_percent = (energy_change_kwh / self.battery_capacity_kwh) * 100.0
        
        # 檢查 SoC 變化是否過大（超過 100% 表示不合理，可能是容量設定錯誤或功率過大）
        # 但我們不限制它，因為可能是合理的（例如快速充放電）
        # 只記錄警告（如果需要）
        
        return soc_change_percent
    
    def should_collect(self, current_time: datetime) -> bool:
        """
        判斷是否應該收集資料
        
        Args:
            current_time: 當前時間
            
        Returns:
            True 如果應該收集，False 否則
        """
        if self.last_collect_time is None:
            return True
        
        elapsed = (current_time - self.last_collect_time).total_seconds()
        return elapsed >= self.collect_interval_sec
    
    def collect(self, 
                battery_data: Dict[str, Tuple[datetime, float, float, float, float, float]],
                mppt_data: Optional[Tuple[float, float, float, float, float, float]],
                initial_soc: float = 50.0,
                battery_count_limit: int = 10,
                vendor_load_count: Optional[int] = None):
        """
        收集資料並寫入 CSV
        
        Args:
            battery_data: 電池資料字典 {battery_id: (ts, soc_pct, volt_v, curr_a, temp_c, speed)}
            mppt_data: MPPT 資料 (solar_v, solar_i, solar_p, mppt_v, mppt_i, mppt_p) 或 None
            initial_soc: 初始 SoC 值（當收到的 SoC 為 0 時使用）
            battery_count_limit: 只記錄前 N 顆電池
            vendor_load_count: 廠商回報的負載顆數（從 Data.txt 第一行解析）
        """
        current_time = datetime.now(TZ_UTC8)
        
        # 檢查是否應該收集
        if not self.should_collect(current_time):
            return False
        
        # 檢查日期是否改變，如果改變則切換到新的 CSV 檔案
        current_date_str = current_time.strftime("%Y-%m-%d")
        if self.current_date != current_date_str:
            self._ensure_csv_file_for_date(current_date_str)
        
        # 解析 MPPT 資料
        solar_v, solar_i, solar_p, mppt_v, mppt_i, mppt_p = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if mppt_data is not None:
            solar_v, solar_i, solar_p, mppt_v, mppt_i, mppt_p = mppt_data
        
        # 計算負載資訊
        lc = vendor_load_count if vendor_load_count is not None else 0
        load_power_w = lc * self.load_power_per_unit_w
        
        # 確保 CSV 檔案已初始化
        if self.csv_file is None:
            self._ensure_csv_file_for_date(current_date_str)
        
        # 寫入每個電池的資料（按照數字順序排序，而非字串順序）
        rows_written = 0
        with open(self.csv_file, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 按照電池 ID 的數字順序排序（例如：1, 2, 3, ..., 10 而非 1, 10, 2, 3）
            def battery_id_sort_key(bid: str) -> int:
                """將電池 ID 轉換為數字用於排序"""
                # 移除可能的 "B" 前綴（如果有）
                bid_clean = bid.lstrip('B')
                try:
                    return int(bid_clean) if bid_clean.isdigit() else 999
                except (ValueError, AttributeError):
                    return 999
            
            # 根據 battery_count_limit 篩選和排序電池
            sorted_battery_ids = sorted(battery_data.keys(), key=battery_id_sort_key)
            
            # 如果沒有電池資料，但至少記錄 MPPT 資料（如果有的話）
            if not sorted_battery_ids:
                if mppt_data is not None:
                    # 至少記錄一次 MPPT 資料（使用當前時間戳）
                    writer.writerow([
                        current_time.strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
                        "",                                          # battery_id (空)
                        "",                                          # soc_percent (空)
                        "",                                          # voltage_v (空)
                        "",                                          # current_ma (空)
                        "",                                          # temp_c (空)
                        "",                                          # speed_percent (空)
                        f"{solar_v:.2f}",                            # solar_v
                        f"{solar_i:.0f}",                            # solar_i_ma (毫安，整數)
                        f"{solar_p:.0f}",                            # solar_p_mw (毫瓦，整數)
                        f"{mppt_v:.2f}",                             # mppt_v
                        f"{mppt_i:.0f}",                             # mppt_i_ma (毫安，整數)
                        f"{mppt_p:.0f}",                             # mppt_p_mw (毫瓦，整數)
                        lc,                                          # load_count
                        f"{load_power_w:.1f}",                       # load_power_w
                    ])
                    rows_written = 1
                    print(f"[資料收集] 記錄了 MPPT 資料（無電池資料）")
                else:
                    print(f"[資料收集] 警告：既沒有電池資料也沒有 MPPT 資料，跳過記錄")
            else:
                # 只取前 battery_count_limit 顆電池
                for battery_id in sorted_battery_ids[:battery_count_limit]:
                    ts, soc_pct, volt_v, curr_a, temp_c, speed = battery_data[battery_id]
                    
                    # 理論計算 SoC 變化（根據老師建議，使用理論計算而非 Data.txt 中的 SoC）
                    battery_key = battery_id.lstrip('B') if battery_id.startswith('B') else battery_id
                    
                    # 如果提供了 Command.txt 路徑，使用理論計算 SoC
                    if self.command_file_path:
                        # 檢查是否有追蹤狀態
                        if battery_key in self.battery_state:
                            last_soc_pct, last_time, last_power_w = self.battery_state[battery_key]
                            
                            # 計算時間間隔（小時）
                            time_interval_sec = (current_time - last_time).total_seconds()
                            time_interval_hours = time_interval_sec / 3600.0
                            
                            # 讀取當前功率命令
                            current_power_w = self._read_command_power(battery_key)
                            if current_power_w is None:
                                # 如果讀不到命令，使用上次的功率（假設持續相同功率）
                                current_power_w = last_power_w
                            
                            # 計算 SoC 變化
                            soc_change = self._calculate_soc_change(current_power_w, time_interval_hours)
                            
                            # 更新 SoC
                            calculated_soc = last_soc_pct + soc_change
                            
                            # 限制 SoC 範圍（0-100%）
                            calculated_soc = max(0.0, min(100.0, calculated_soc))
                            
                            # 使用理論計算的 SoC
                            soc_pct = calculated_soc
                            
                            # 更新追蹤狀態
                            self.battery_state[battery_key] = (calculated_soc, current_time, current_power_w)
                        else:
                            # 第一次記錄，使用初始 SoC（如果 Data.txt 中的 SoC 為 0，使用 initial_soc）
                            if soc_pct == 0.0:
                                soc_pct = initial_soc
                            
                            # 讀取當前功率命令
                            current_power_w = self._read_command_power(battery_key)
                            if current_power_w is None:
                                current_power_w = 0.0
                            
                            # 初始化追蹤狀態
                            self.battery_state[battery_key] = (soc_pct, current_time, current_power_w)
                    else:
                        # 如果沒有提供 Command.txt 路徑，使用 Data.txt 中的 SoC（但 SoC 為 0 時使用 initial_soc）
                        if soc_pct == 0.0:
                            soc_pct = initial_soc
                    
                    # 寫入一行
                    writer.writerow([
                        ts.strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
                        battery_id,                         # battery_id
                        f"{soc_pct:.2f}",                   # soc_percent
                        f"{volt_v:.2f}",                    # voltage_v
                        f"{curr_a:.0f}",                    # current_ma (毫安，整數)
                        f"{temp_c:.1f}",                    # temp_c
                        f"{speed:.1f}",                     # speed_percent
                        f"{solar_v:.2f}",                   # solar_v
                        f"{solar_i:.0f}",                   # solar_i_ma (毫安，整數)
                        f"{solar_p:.0f}",                   # solar_p_mw (毫瓦，整數)
                        f"{mppt_v:.2f}",                    # mppt_v
                        f"{mppt_i:.0f}",                    # mppt_i_ma (毫安，整數)
                        f"{mppt_p:.0f}",                    # mppt_p_mw (毫瓦，整數)
                        lc,                                 # load_count
                        f"{load_power_w:.1f}",              # load_power_w
                    ])
                    rows_written += 1
        
        # 更新最後收集時間
        self.last_collect_time = current_time
        
        return rows_written > 0
    
    def get_collected_count(self) -> int:
        """
        取得已收集的資料筆數（所有日期的 CSV 檔案總和）
        
        Returns:
            總資料筆數
        """
        total_count = 0
        
        # 掃描輸出目錄中所有符合格式的 CSV 檔案
        if not os.path.exists(self.output_dir):
            return 0
        
        try:
            for filename in os.listdir(self.output_dir):
                if filename.startswith("collected_data_") and filename.endswith(".csv"):
                    csv_path = os.path.join(self.output_dir, filename)
                    try:
                        with open(csv_path, 'r', encoding='utf-8-sig') as f:
                            # 減去標題行
                            file_count = sum(1 for _ in f) - 1
                            if file_count > 0:
                                total_count += file_count
                    except Exception:
                        continue
        except Exception:
            pass
        
        return total_count

