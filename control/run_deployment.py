#!/usr/bin/env python3
"""
P302 即時部署控制迴圈
====================
完整流程：
  1. 每 ~10 秒讀取 Data.txt（韌體每 10-12 秒更新一次）
  2. 將 MPPT / 電池讀數放入 15 分鐘 buffer
  3. 每 15 分鐘聚合 buffer → 計算狀態向量 → 模型推論 → 寫 Command.txt
  4. 中間每秒更新 Command.txt 時間戳（維持與廠商軟體同步）

Data Flow:
  ┌──────────┐    Data.txt    ┌────────────────┐
  │ Hardware  │ ───(10-12s)──→│  DataBuffer     │
  └──────────┘                │  (15min window) │
                              └───────┬────────┘
                                      │ every 15min
                              ┌───────▼────────┐
                              │  Aggregation    │
                              │  mean/std/max   │
                              └───────┬────────┘
                                      │
                              ┌───────▼────────┐
                              │  State Builder  │
                              │  [SoC, load,    │
                              │   pv, price,    │
                              │   hour, dow]    │
                              └───────┬────────┘
                                      │
                              ┌───────▼────────┐
                              │  SAC Agent      │
                              │  → action       │
                              └───────┬────────┘
                                      │
  ┌──────────┐  Command.txt   ┌───────▼────────┐
  │ Hardware  │ ←──(1s)──────│  Command Writer │
  └──────────┘                └────────────────┘

單位約定（P302 鋅空氣電池）：
  - MPPT_P in Data.txt : mW（800 = 800 mW = 0.8 W）
  - power in Command.txt : mW（170 = 170 mW = 0.17 W）
  - flow in Command.txt  : %（25 = 25%）
  - 模型內部使用 kW

用法：
  python control/run_deployment.py ^
      --data-file ./Data.txt ^
      --command-file ./Command.txt ^
      --model-path ./models/best_sac_model.pth ^
      --battery-id 01
"""

import os
import sys
import time
import math
import argparse
import json
import csv
from datetime import datetime, timezone, timedelta, time as dtime
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

# ── 路徑設定 ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core'))
sys.path.insert(0, SCRIPT_DIR)

from sac_agent import SACAgent
from io_protocol import (
    read_vendor_data_file,
    write_control_file_vendor,
    format_ts,
    parse_ts,
    TZ_UTC8,
)

# ══════════════════════════════════════════════════════════════════
# 常數（P302 鋅空氣電池規格）
# ══════════════════════════════════════════════════════════════════
# 模擬 4 組模組並聯之容量（依據 8Feb2025 規劃書）
# 容量: 1.4792A × 2hr × 4組 = 11833.6 mAh, 16.57Wh × 4 = 66.28 Wh
# 功率: 維持 P302 硬體物理極限 20mA × 8.5V = 170mW
BATTERY_CAPACITY_MAH  = 11833.6     # 4組總容量 (mAh)
BATTERY_CHARGE_V      = 8.5         # V
BATTERY_DISCHARGE_V   = 5.6         # V
BATTERY_CAPACITY_WH   = 66.28       # 4組總儲能 (Wh)
BATTERY_CAPACITY_KWH  = BATTERY_CAPACITY_WH / 1000   # ≈ 0.06628 kWh
BATTERY_PMAX_KW       = 0.00017     # P302 硬體極限 170mW (20mA × 8.5V)
BATTERY_EFFICIENCY    = 0.85

# 負載規格
LOAD_PER_GROUP_W = 8.0    # W per group (vendor confirmed)
MAX_LOAD_GROUPS  = 4

# 負載時刻表（0-4 組，每組 12W @5V）
LOAD_SCHEDULE = [
    (dtime( 0, 0), 0), (dtime( 1, 0), 0), (dtime( 2, 0), 0),
    (dtime( 3, 0), 0), (dtime( 4, 0), 0), (dtime( 5, 0), 0),
    (dtime( 6, 0), 1), (dtime( 7, 0), 1), (dtime( 8, 0), 2),
    (dtime( 9, 0), 3), (dtime(10, 0), 4), (dtime(11, 0), 4),
    (dtime(12, 0), 4), (dtime(13, 0), 4), (dtime(14, 0), 4),
    (dtime(15, 0), 3), (dtime(16, 0), 3), (dtime(17, 0), 2),
    (dtime(18, 0), 2), (dtime(19, 0), 1), (dtime(20, 0), 1),
    (dtime(21, 0), 0), (dtime(22, 0), 0), (dtime(23, 0), 0),
]

# ── 2026 台電夏季 TOU 電價 (TWD/kWh) ──────────────────────────
# 平日：離峰 00-09 = 2.06, 半尖峰 09-16 = 4.69,
#        尖峰 16-22 = 7.13, 晚半尖峰 22-24 = 4.69
# 週末/假日：全天 2.06（無套利空間）
TOU_OFFPEAK  = 2.06   # TWD/kWh
TOU_MIDPEAK  = 4.69
TOU_PEAK     = 7.13


def get_load_groups(t: dtime) -> int:
    """根據時間查表回傳負載組數"""
    result = 0
    for sched_t, n in LOAD_SCHEDULE:
        if t >= sched_t:
            result = n
    return result


def get_tou_price(hour: int, day_of_week: int = 0) -> float:
    """
    2026 台電夏季 TOU 電價。
    
    Args:
        hour: 0~23
        day_of_week: 0=Mon ... 6=Sun
    
    Returns:
        電價 (TWD/kWh)
    """
    # 週末全天離峰
    if day_of_week >= 5:  # Sat=5, Sun=6
        return TOU_OFFPEAK
    # 平日
    if hour < 9:
        return TOU_OFFPEAK      # 00:00 - 09:00
    elif hour < 16:
        return TOU_MIDPEAK      # 09:00 - 16:00
    elif hour < 22:
        return TOU_PEAK         # 16:00 - 22:00
    else:
        return TOU_MIDPEAK      # 22:00 - 24:00


# ══════════════════════════════════════════════════════════════════
# 讀取紀錄（單次 Data.txt 快照）
# ══════════════════════════════════════════════════════════════════
@dataclass
class Reading:
    """Data.txt 的一次讀取紀錄"""
    timestamp: datetime
    # MPPT（太陽能）
    mppt_p_mw: float = 0.0     # 功率 mW（800 = 800 mW）
    mppt_v: float = 0.0        # 電壓 V
    mppt_i_ma: float = 0.0     # 電流 mA
    solar_p_mw: float = 0.0    # Raw solar power mW
    # MPPT-Bus（新格式 2026/03）
    bus_v: float = 0.0         # Bus 電壓 V
    bus_i_ma: float = 0.0      # Bus 電流 mA
    bus_p_mw: float = 0.0      # Bus 功率 mW
    # 負載實測（新格式 2026/03）
    load_v: float = 0.0        # 負載電壓 V
    load_i_ma: float = 0.0     # 負載電流 mA
    load_p_mw: float = 0.0     # 負載功率 mW
    # 電池
    batt_soc_pct: float = 50.0 # 韌體 SoC %（參考用）
    batt_v: float = 0.0        # 電池電壓 V
    batt_i_ma: float = 0.0     # 電池電流 mA
    batt_temp_c: float = 25.0  # 溫度 °C
    batt_speed_pct: float = 0.0  # 流速 %


# ══════════════════════════════════════════════════════════════════
# 15 分鐘 Buffer + 聚合
# ══════════════════════════════════════════════════════════════════
class DataBuffer:
    """
    收集 15 分鐘窗格內的所有 Data.txt 讀取，
    並在窗格結束時聚合成模型輸入。
    """

    def __init__(self, window_sec: int = 900):
        self.window_sec = window_sec   # 15 min = 900 sec
        self.readings: List[Reading] = []
        self.window_start: Optional[datetime] = None

    def add(self, reading: Reading):
        """加入一筆讀取"""
        now = reading.timestamp
        if self.window_start is None:
            # 對齊到 15 分鐘邊界
            minute = now.minute
            aligned_min = (minute // 15) * 15
            self.window_start = now.replace(minute=aligned_min, second=0, microsecond=0)
        self.readings.append(reading)

    def is_window_complete(self, now: datetime) -> bool:
        """檢查目前窗格是否已滿 15 分鐘"""
        if self.window_start is None:
            return False
        elapsed = (now - self.window_start).total_seconds()
        return elapsed >= self.window_sec

    def aggregate(self) -> Dict[str, float]:
        """
        聚合窗格內所有讀取，回傳統計值。
        所有功率統一用 mW，需要 kW 時在使用處轉換。
        
        Returns:
            {
                'mppt_p_mean_mW': float,   # MPPT 平均功率 (mW)
                'mppt_p_std_mW': float,    # MPPT 標準差 (mW)
                'mppt_p_max_mW': float,    # MPPT 最大值 (mW)
                'bus_p_mean_mW': float,    # MPPT-Bus 平均功率 (mW)
                'load_p_mean_mW': float,   # 負載實測平均功率 (mW)
                'batt_p_mean_mW': float,   # 電池平均功率 (mW) = mean(V_i × I_i)
                'batt_v_mean': float,      # 電池電壓平均 (V) — SoCTracker 診斷用
                'batt_i_mean_ma': float,   # 電池電流平均 (mA) — SoCTracker 診斷用
                'n_samples': int,          # 讀取數量
                'completeness': float,     # 完整度 (0~1)
            }
        """
        if not self.readings:
            return {
                'mppt_p_mean_mW': 0.0, 'mppt_p_std_mW': 0.0, 'mppt_p_max_mW': 0.0,
                'bus_p_mean_mW': 0.0,
                'load_p_mean_mW': 0.0,
                'batt_p_mean_mW': 0.0,
                'batt_v_mean': 0.0, 'batt_i_mean_ma': 0.0,
                'n_samples': 0, 'completeness': 0.0,
            }

        mppt_vals = [r.mppt_p_mw for r in self.readings]
        # 電池功率：正確計算 mean(V_i × I_i)，而非 mean(V) × mean(I)
        batt_p_vals = [r.batt_v * r.batt_i_ma for r in self.readings]  # V × mA = mW
        batt_v_vals = [r.batt_v for r in self.readings]
        batt_i_vals = [r.batt_i_ma for r in self.readings]

        # 新格式欄位（可能為 0 如果用舊格式韌體）
        bus_p_vals = [r.bus_p_mw for r in self.readings if r.bus_p_mw > 0]
        load_p_vals = [r.load_p_mw for r in self.readings if r.load_p_mw > 0]

        n = len(self.readings)
        # 預期 15min / 11s ≈ 82 筆
        expected = self.window_sec / 11.0
        completeness = min(n / expected, 1.0) if expected > 0 else 0.0

        mppt_mean = float(np.mean(mppt_vals)) if mppt_vals else 0.0
        mppt_std = float(np.std(mppt_vals)) if len(mppt_vals) > 1 else 0.0
        mppt_max = float(np.max(mppt_vals)) if mppt_vals else 0.0

        bus_p_mean = float(np.mean(bus_p_vals)) if bus_p_vals else 0.0
        load_p_mean = float(np.mean(load_p_vals)) if load_p_vals else 0.0

        return {
            'mppt_p_mean_mW': mppt_mean,
            'mppt_p_std_mW': mppt_std,
            'mppt_p_max_mW': mppt_max,
            'bus_p_mean_mW': bus_p_mean,
            'load_p_mean_mW': load_p_mean,
            'batt_p_mean_mW': float(np.mean(batt_p_vals)) if batt_p_vals else 0.0,
            'batt_v_mean': float(np.mean(batt_v_vals)) if batt_v_vals else 0.0,
            'batt_i_mean_ma': float(np.mean(batt_i_vals)) if batt_i_vals else 0.0,
            'n_samples': n,
            'completeness': completeness,
        }

    def reset(self, new_start: Optional[datetime] = None):
        """清空 buffer，開始新窗格"""
        self.readings.clear()
        self.window_start = new_start

    @property
    def count(self) -> int:
        return len(self.readings)


# ══════════════════════════════════════════════════════════════════
# SoC 自計算（庫倫計數法 — Coulomb Counting, Ah 基準）
# ══════════════════════════════════════════════════════════════════
class SoCTracker:
    """
    基於庫倫計數 (Ah) 的 SoC 追蹤器。
    
    不依賴韌體 SoC，使用電池電流自行估算。
    
    方法：標準庫倫積分（只積電流，不乘電壓）
        ΔSoC = (I_ma × Δt_h × η_coulombic) / capacity_mah
    
    效率定義：
        BATTERY_EFFICIENCY = 0.85 → Round-Trip Efficiency (RTE)
        單向庫倫效率 η = √(RTE) ≈ 0.922
        - 充電 (I > 0): 存入電荷 = 量測電荷 × η（化學轉換損耗）
        - 放電 (I < 0): 取出電荷 = 量測電荷 / η（內阻損耗）
    
    注意：
        - voltage_v 參數保留但僅用於統計/日誌，不參與 SoC 計算
        - 若需要「能量法 SoC」（考慮電壓變化），請使用 SoCTrackerEnergy
    """

    # 長斷線處理：超過此秒數仍會計算，但會 clamp 到此上限
    MAX_INTEGRATION_SEC = 3600.0  # 1 小時

    def __init__(self, initial_soc: float = 0.5,
                 capacity_mah: float = BATTERY_CAPACITY_MAH,
                 efficiency_rte: float = BATTERY_EFFICIENCY):
        self.soc = initial_soc
        self.capacity_mah = capacity_mah
        # 單向庫倫效率 = sqrt(RTE)
        self.eta = float(np.sqrt(max(efficiency_rte, 0.01)))
        self.last_update: Optional[datetime] = None

        # 統計用
        self.total_charge_mah = 0.0
        self.total_discharge_mah = 0.0
        self.skipped_intervals = 0  # 被截斷/跳過的異常間隔計數

    def update(self, timestamp: datetime, current_ma: float, voltage_v: float = 0.0):
        """
        根據電流與時間更新 SoC（標準庫倫計數）。
        
        Args:
            timestamp: 當前時間
            current_ma: 電池電流 (mA)，正=充電, 負=放電
            voltage_v: 電池電壓 (V)，僅記錄用，不影響 SoC 計算
        """
        if self.last_update is None:
            self.last_update = timestamp
            return

        dt_sec = (timestamp - self.last_update).total_seconds()
        self.last_update = timestamp

        # 時間倒退：跳過
        if dt_sec <= 0:
            return

        # 長斷線處理：仍然計算，但截斷到上限（避免 sensor 累積偏差放大）
        if dt_sec > self.MAX_INTEGRATION_SEC:
            self.skipped_intervals += 1
            dt_sec = self.MAX_INTEGRATION_SEC  # 截斷而非丟棄

        dt_h = dt_sec / 3600.0

        # 庫倫積分 (mAh) — 只看電流，不乘電壓
        delta_mah_raw = current_ma * dt_h  # mA × h = mAh

        if current_ma > 0:
            # 充電：實際存入 = 量測 × η（部分能量損耗於化學轉換）
            effective_mah = delta_mah_raw * self.eta
            self.total_charge_mah += effective_mah
        elif current_ma < 0:
            # 放電：實際減少 = 量測 / η（內阻使內部消耗 > 外部量測）
            effective_mah = delta_mah_raw / self.eta  # 負值
            self.total_discharge_mah += abs(effective_mah)
        else:
            effective_mah = 0.0

        # 更新 SoC
        if self.capacity_mah > 0:
            delta_soc = effective_mah / self.capacity_mah
            self.soc = float(np.clip(self.soc + delta_soc, 0.0, 1.0))

    def update_from_buffer(self, readings: List[Reading]):
        """用整段 buffer 更新 SoC（用於 15 分鐘聚合前）"""
        for r in readings:
            self.update(r.timestamp, r.batt_i_ma, r.batt_v)

    def get_soc(self) -> float:
        return self.soc

    def set_soc(self, soc: float):
        """外部校正 SoC（例如從已知狀態重置）"""
        self.soc = float(np.clip(soc, 0.0, 1.0))

    def get_stats(self) -> Dict[str, float]:
        """回傳統計資訊"""
        return {
            'soc': self.soc,
            'total_charge_mah': self.total_charge_mah,
            'total_discharge_mah': self.total_discharge_mah,
            'skipped_intervals': self.skipped_intervals,
            'eta_coulombic': self.eta,
        }


# ══════════════════════════════════════════════════════════════════
# 狀態建構
# ══════════════════════════════════════════════════════════════════
def build_state_from_aggregation(
    agg: Dict[str, float],
    soc: float,
    now: datetime,
) -> np.ndarray:
    """
    從 15 分鐘聚合數據建構 6D 狀態向量。
    
    state = [SoC, load_kW, pv_kW, price_norm, hour, day_of_week]
    
    Args:
        agg: DataBuffer.aggregate() 的輸出
        soc: 當前 SoC (0~1)
        now: 當前時間
    
    Returns:
        6D numpy array (float32)
    """
    # PV: 從 MPPT 聚合數據 (mW → kW)
    pv_kw = agg['mppt_p_mean_mW'] / 1e6  # mW → kW

    # 負載: 優先用 Data.txt 實測值（新格式），否則用排程估計
    load_measured_kw = agg.get('load_p_mean_mW', 0.0) / 1e6  # mW → kW
    if load_measured_kw > 0:
        load_kw = load_measured_kw   # 使用實測值
    else:
        # 舊格式 fallback: 從時間表查詢
        load_groups = get_load_groups(now.time())
        load_w = load_groups * LOAD_PER_GROUP_W
        load_kw = load_w / 1000.0

    # 電價（2026 台電 TOU）
    price = get_tou_price(now.hour, now.weekday())
    price_norm = float(np.clip(price / 10.0, 0.0, 1.0))  # 正規化：7.13 TWD → ~0.71

    # 時間特徵
    hour = float(now.hour)
    dow = float(now.weekday())  # 0=Mon, 6=Sun

    state = np.array([
        soc,         # 0: SoC (0~1)
        load_kw,     # 1: 負載 (kW)
        pv_kw,       # 2: PV (kW) — MPPT 15min 平均
        price_norm,  # 3: 電價（正規化）
        hour,        # 4: 小時
        dow,         # 5: 星期
    ], dtype=np.float32)

    return state


# ══════════════════════════════════════════════════════════════════
# 情況碼判定
# ══════════════════════════════════════════════════════════════════
def determine_situation(action_kw: float, load_kw: float, pv_kw: float) -> int:
    """
    判定能流情況碼（1~4）。
    
    前提：太陽能優先供負載，不逆送市電。
    剩餘負載 = max(0, load - pv)
    
    情況碼：
      1: 電池放電全包剩餘負載
      2: 電池放電 + 市電補足
      3: 市電同時供負載及充電
      4: 電池待機（100% 市電供剩餘負載）
    """
    net_load = max(0.0, load_kw - pv_kw)

    if action_kw < -0.0001:  # 放電
        discharge_kw = abs(action_kw)
        if discharge_kw >= net_load - 0.0001:
            return 1
        else:
            return 2
    elif action_kw > 0.0001:  # 充電
        return 3
    else:
        return 4


# ══════════════════════════════════════════════════════════════════
# 模型載入
# ══════════════════════════════════════════════════════════════════
def load_agent(model_path: str, state_dim: int = 6, action_dim: int = 2,
               device: str = "cpu") -> SACAgent:
    """載入 SAC Agent，自動推斷網路結構"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    hidden_dim = 128  # 預設
    if 'actor' in checkpoint:
        actor_state = checkpoint['actor']
        if 'fc1.weight' in actor_state:
            hidden_dim = actor_state['fc1.weight'].shape[0]
            # 自動推斷 action_dim
            if 'log_std_layer.weight' in actor_state:
                action_dim = actor_state['log_std_layer.weight'].shape[0]
            elif 'fc_mean.weight' in actor_state:
                action_dim = actor_state['fc_mean.weight'].shape[0]

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        hidden_dim=hidden_dim,
    )
    agent.load(model_path)
    agent.device = device
    print(f"  模型載入成功: state_dim={state_dim}, action_dim={action_dim}, "
          f"hidden_dim={hidden_dim}")
    return agent, action_dim


# ══════════════════════════════════════════════════════════════════
# Data.txt 讀取（不清空，韌體持續覆寫）
# ══════════════════════════════════════════════════════════════════
def read_data_txt(path: str, battery_pp: str = "01") -> Optional[Reading]:
    """
    讀取一次 Data.txt，解析為 Reading。
    
    不清空檔案（clear_after_read=False），因為韌體持續覆寫。
    
    Returns:
        Reading 或 None（讀取失敗/檔案不存在）
    """
    try:
        result = read_vendor_data_file(
            path, max_age_sec=60, clear_after_read=False
        )
        mppt_data = result['mppt']
        batt_data = result['batteries']
        mppt_bus = result.get('mppt_bus')   # 新格式才有
        load_hw  = result.get('load')       # 新格式才有
    except Exception:
        return None

    now = datetime.now(TZ_UTC8)
    reading = Reading(timestamp=now)

    # MPPT 數據
    if mppt_data is not None:
        solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw = mppt_data
        reading.mppt_p_mw = mppt_p_mw      # 已是 mW（800 = 800 mW）
        reading.mppt_v = mppt_v
        reading.mppt_i_ma = mppt_i_ma
        reading.solar_p_mw = solar_p_mw

    # MPPT-Bus 數據（新格式）
    if mppt_bus is not None:
        reading.bus_v = mppt_bus[0]
        reading.bus_i_ma = mppt_bus[1]
        reading.bus_p_mw = mppt_bus[2]

    # 負載實測數據（新格式）
    if load_hw is not None:
        reading.load_v = load_hw[0]
        reading.load_i_ma = load_hw[1]
        reading.load_p_mw = load_hw[2]

    # 電池數據
    if battery_pp in batt_data:
        ts, soc_pct, volt_v, curr_ma, temp_c, speed = batt_data[battery_pp]
        reading.batt_soc_pct = soc_pct
        reading.batt_v = volt_v
        reading.batt_i_ma = curr_ma
        reading.batt_temp_c = temp_c
        reading.batt_speed_pct = speed
        # 用檔案裡的時間戳（更準確）
        if ts is not None:
            reading.timestamp = ts

    return reading


# ══════════════════════════════════════════════════════════════════
# CSV 日誌
# ══════════════════════════════════════════════════════════════════
class DeploymentLogger:
    """記錄每個 15 分鐘決策到 CSV"""

    HEADER = [
        'timestamp', 'step',
        'soc', 'load_kw', 'pv_kw', 'price', 'hour', 'dow',
        'mppt_mean_mW', 'mppt_max_mW', 'mppt_std_mW',
        'bus_p_mean_mW', 'load_p_mean_mW',
        'batt_p_mean_mW', 'batt_v_mean', 'batt_i_mean_ma',
        'n_samples', 'completeness',
        'action_power_kw', 'action_flow_pct',
        'power_mw_cmd', 'flow_pct_cmd', 'situation_code',
        'load_groups',
    ]

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(log_dir, f'deployment_{date_str}.csv')
        with open(self.path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER)
        print(f"  日誌: {self.path}")

    def log(self, row: Dict[str, Any]):
        with open(self.path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([row.get(k, '') for k in self.HEADER])


# ══════════════════════════════════════════════════════════════════
# 主控制迴圈
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="P302 即時部署控制迴圈（15 分鐘聚合 + SAC 推論）")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Data.txt 路徑（韌體寫入）")
    parser.add_argument("--command-file", type=str, required=True,
                        help="Command.txt 路徑（AI 輸出）")
    parser.add_argument("--model-path", type=str, required=True,
                        help="SAC 模型 .pth 路徑")
    parser.add_argument("--battery-pp", type=str, default="01",
                        help="目標電池 PP 編號（01-10）")
    parser.add_argument("--initial-soc", type=float, default=0.5,
                        help="初始 SoC (0~1)")
    parser.add_argument("--poll-sec", type=float, default=10.0,
                        help="Data.txt 輪詢間隔（秒），預設 10")
    parser.add_argument("--window-min", type=int, default=15,
                        help="聚合窗格（分鐘），預設 15")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--log-dir", type=str, default=None,
                        help="日誌輸出目錄（預設: results/deployment/）")
    parser.add_argument("--dry-run", action="store_true",
                        help="乾跑模式：不寫 Command.txt，僅印出決策")
    parser.add_argument("--initial-action", type=str, default="standby",
                        choices=["standby", "random"],
                        help="首次 15 分鐘的動作（standby=待機, random=隨機）")
    args = parser.parse_args()

    # ── 初始化 ──────────────────────────────────────────────
    print("=" * 70)
    print("  P302 即時部署控制迴圈")
    print("  電池: 10mAh / 20mA / 8.5V(充) / 5.6V(放)")
    print("  負載: 4 組 × 12W @5V")
    print("=" * 70)

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    agent, action_dim = load_agent(args.model_path, state_dim=6, device=device)
    is_2d = (action_dim >= 2)
    print(f"  Action 維度: {action_dim} ({'power+flow' if is_2d else 'power only'})")

    soc_tracker = SoCTracker(initial_soc=args.initial_soc)
    buffer = DataBuffer(window_sec=args.window_min * 60)

    log_dir = args.log_dir or os.path.join(PROJECT_ROOT, 'results', 'deployment')
    logger = DeploymentLogger(log_dir)

    pp = f"{int(args.battery_pp):02d}"
    window_sec = args.window_min * 60
    step_count = 0

    # 上一次動作（用於持續更新 Command.txt）
    # 注意：write_control_file_vendor 期望 power 單位為 W（內部 ×1000 轉 mW）
    last_power_w = 0.0
    last_flow_pct = 0.0
    last_sit_code = 4  # 待機
    last_command_write: Optional[datetime] = None

    print(f"\n  Data.txt  : {args.data_file}")
    print(f"  Command   : {args.command_file}")
    print(f"  模型      : {args.model_path}")
    print(f"  電池 PP   : {pp}")
    print(f"  初始 SoC  : {args.initial_soc * 100:.0f}%")
    print(f"  輪詢間隔  : {args.poll_sec}s")
    print(f"  聚合窗格  : {args.window_min} min")
    print(f"  首次動作  : {args.initial_action}")
    print(f"  乾跑模式  : {args.dry_run}")
    print()
    print("-" * 70)
    print(f"  開始時間: {datetime.now(TZ_UTC8).strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    # ── 首次動作（第一個 15 分鐘用隨機或待機）──────────────
    if args.initial_action == "random":
        # 隨機小功率動作
        rnd_power_kw = float(np.random.uniform(-BATTERY_PMAX_KW, BATTERY_PMAX_KW))
        rnd_flow_pct = float(np.random.uniform(10, 50))
        last_power_w = abs(rnd_power_kw) * 1e3  # kW → W
        last_flow_pct = rnd_flow_pct
        last_sit_code = determine_situation(rnd_power_kw, 0.0, 0.0)
        print(f"  [初始] 隨機動作: power={last_power_w*1000:.1f}mW ({last_power_w:.4f}W), flow={last_flow_pct:.0f}%")
    else:
        print(f"  [初始] 待機: power=0mW, flow=0%")

    # ── 主迴圈 ──────────────────────────────────────────────
    try:
        while True:
            loop_start = time.time()
            now = datetime.now(TZ_UTC8)

            # ── A) 讀取 Data.txt ──────────────────────────
            reading = read_data_txt(args.data_file, battery_pp=pp)
            if reading is not None:
                buffer.add(reading)
                # 即時更新 SoC
                soc_tracker.update(
                    reading.timestamp, reading.batt_i_ma, reading.batt_v
                )

                if buffer.count % 10 == 1:  # 每 10 筆印一次
                    ts_str = reading.timestamp.strftime('%H:%M:%S')
                    print(f"  [{ts_str}] 讀取 #{buffer.count:3d}  "
                          f"MPPT={reading.mppt_p_mw:6.0f}mW  "
                          f"V={reading.batt_v:.2f}V  I={reading.batt_i_ma:.0f}mA  "
                          f"SoC={soc_tracker.get_soc()*100:.1f}%  "
                          f"(fw:{reading.batt_soc_pct:.1f}%)")
            else:
                ts_str = now.strftime('%H:%M:%S')
                if buffer.count == 0:
                    print(f"  [{ts_str}] 等待 Data.txt...")

            # ── B) 檢查是否到 15 分鐘邊界 ────────────────
            if buffer.is_window_complete(now) and buffer.count > 0:
                step_count += 1
                print(f"\n{'='*60}")
                print(f"  [Step {step_count}] 15 分鐘聚合 ({buffer.count} 筆)")
                print(f"{'='*60}")

                # 聚合
                agg = buffer.aggregate()
                soc = soc_tracker.get_soc()

                mppt_mw = agg['mppt_p_mean_mW']
                print(f"  MPPT 平均: {mppt_mw:.1f} mW ({mppt_mw/1000:.4f} W)")
                print(f"  MPPT 最大: {agg['mppt_p_max_mW']:.1f} mW  標準差: {agg['mppt_p_std_mW']:.1f} mW")
                if agg['bus_p_mean_mW'] > 0:
                    print(f"  MPPT-Bus: {agg['bus_p_mean_mW']:.1f} mW")
                if agg['load_p_mean_mW'] > 0:
                    print(f"  負載實測: {agg['load_p_mean_mW']:.1f} mW")
                print(f"  電池功率: {agg['batt_p_mean_mW']:.1f} mW "
                      f"(V={agg['batt_v_mean']:.2f}V, I={agg['batt_i_mean_ma']:.1f}mA)")
                print(f"  完整度: {agg['completeness']*100:.0f}% ({agg['n_samples']} 筆)")
                print(f"  SoC (自算): {soc*100:.1f}%")

                # 建構狀態
                state = build_state_from_aggregation(agg, soc, now)
                load_groups = get_load_groups(now.time())
                load_kw = state[1]
                pv_kw = state[2]
                price = state[3]

                print(f"\n  State: SoC={state[0]:.3f}, Load={load_kw*1000:.1f}mW({load_groups}組), "
                      f"PV={pv_kw*1e6:.1f}mW, Price={price:.2f}, "
                      f"Hour={int(state[4])}, DoW={int(state[5])}")

                # ── C) 模型推論 ──────────────────────────
                with torch.no_grad():
                    action_norm = agent.select_action(state, evaluate=True)

                if is_2d:
                    power_norm = float(action_norm[0])   # [-1, 1]
                    flow_norm = float(action_norm[1])    # [0, 1] or [-1, 1]
                    # flow 映射到 [0, 100]%
                    flow_pct = float(np.clip((flow_norm + 1) / 2 * 100, 0, 100))
                else:
                    power_norm = float(action_norm[0])
                    # 1D: 用功率比例推算流速
                    flow_pct = abs(power_norm) * 100.0

                # 功率 kW（正=充電, 負=放電）
                action_kw = power_norm * BATTERY_PMAX_KW

                # SoC 安全檢查
                if soc <= 0.10 and action_kw < 0:
                    action_kw = 0.0
                    print(f"  ⚠ SoC 過低 ({soc*100:.1f}%)，禁止放電")
                elif soc >= 0.95 and action_kw > 0:
                    action_kw = 0.0
                    print(f"  ⚠ SoC 過高 ({soc*100:.1f}%)，禁止充電")

                # 情況碼
                sit_code = determine_situation(action_kw, load_kw, pv_kw)

                # Command.txt 輸出值（W 單位，write_control_file_vendor 內部 ×1000 轉 mW）
                power_w = abs(action_kw) * 1e3  # kW → W
                power_mw_display = power_w * 1000.0  # 顯示用 mW
                direction = "充電" if action_kw > 0.0001 else ("放電" if action_kw < -0.0001 else "待機")

                print(f"\n  決策: 情況{sit_code}({direction})")
                print(f"    功率: {power_mw_display:.1f} mW = {power_w:.4f} W")
                print(f"    流速: {flow_pct:.0f}%")
                print(f"    raw action: {action_norm}")

                # 更新記憶
                last_power_w = power_w
                last_flow_pct = flow_pct
                last_sit_code = sit_code

                # ── D) 寫 Command.txt ────────────────────
                if not args.dry_run:
                    write_ts = datetime.now(TZ_UTC8)
                    success = write_control_file_vendor(
                        args.command_file,
                        {pp: (write_ts, power_w, flow_pct)},
                        global_ts=write_ts,
                        require_empty=True,
                        max_wait_sec=0.2,
                        max_retries=5,
                        situation_code=sit_code,
                    )
                    if not success:
                        success = write_control_file_vendor(
                            args.command_file,
                            {pp: (write_ts, power_w, flow_pct)},
                            global_ts=write_ts,
                            require_empty=False,
                            max_wait_sec=0.1,
                            max_retries=3,
                            situation_code=sit_code,
                        )
                    if success:
                        last_command_write = write_ts
                        print(f"  ✓ Command.txt 已更新")
                    else:
                        print(f"  ✗ Command.txt 寫入失敗")
                else:
                    print(f"  [DRY RUN] 跳過 Command.txt 寫入")

                # ── E) 日誌 ──────────────────────────────
                logger.log({
                    'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                    'step': step_count,
                    'soc': f'{soc:.4f}',
                    'load_kw': f'{load_kw:.6f}',
                    'pv_kw': f'{pv_kw:.8f}',
                    'price': f'{price:.3f}',
                    'hour': int(state[4]),
                    'dow': int(state[5]),
                    'mppt_mean_mW': f'{agg["mppt_p_mean_mW"]:.2f}',
                    'mppt_max_mW': f'{agg["mppt_p_max_mW"]:.2f}',
                    'mppt_std_mW': f'{agg["mppt_p_std_mW"]:.2f}',
                    'bus_p_mean_mW': f'{agg["bus_p_mean_mW"]:.2f}',
                    'load_p_mean_mW': f'{agg["load_p_mean_mW"]:.2f}',
                    'batt_p_mean_mW': f'{agg["batt_p_mean_mW"]:.2f}',
                    'batt_v_mean': f'{agg["batt_v_mean"]:.3f}',
                    'batt_i_mean_ma': f'{agg["batt_i_mean_ma"]:.1f}',
                    'n_samples': agg['n_samples'],
                    'completeness': f'{agg["completeness"]:.3f}',
                    'action_power_kw': f'{action_kw:.8f}',
                    'action_flow_pct': f'{flow_pct:.1f}',
                    'power_mw_cmd': f'{power_mw_display:.1f}',
                    'flow_pct_cmd': f'{flow_pct:.1f}',
                    'situation_code': sit_code,
                    'load_groups': load_groups,
                })

                # ── F) 重置 buffer ────────────────────────
                # 對齊到下一個 15 分鐘邊界
                next_min = ((now.minute // args.window_min) + 1) * args.window_min
                if next_min >= 60:
                    next_start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    next_start = now.replace(minute=next_min, second=0, microsecond=0)
                buffer.reset(new_start=next_start)
                print(f"\n  下一個窗格: {next_start.strftime('%H:%M:%S')}")
                print("-" * 60)

            # ── 中間：每秒更新 Command.txt 時間戳 ────────
            elif not args.dry_run:
                now_ts = datetime.now(TZ_UTC8)
                should_update = (
                    last_command_write is None or
                    (now_ts - last_command_write).total_seconds() >= 1.0
                )
                if should_update:
                    success = write_control_file_vendor(
                        args.command_file,
                        {pp: (now_ts, last_power_w, last_flow_pct)},
                        global_ts=now_ts,
                        require_empty=True,
                        max_wait_sec=0.1,
                        max_retries=2,
                        situation_code=last_sit_code,
                    )
                    if not success:
                        write_control_file_vendor(
                            args.command_file,
                            {pp: (now_ts, last_power_w, last_flow_pct)},
                            global_ts=now_ts,
                            require_empty=False,
                            max_wait_sec=0.05,
                            max_retries=1,
                            situation_code=last_sit_code,
                        )
                    last_command_write = now_ts

            # ── 等待下次輪詢 ────────────────────────────
            elapsed = time.time() - loop_start
            sleep_time = max(0.5, args.poll_sec - elapsed)
            # 但如果快到窗格邊界了，縮短等待
            if buffer.window_start is not None:
                remaining = (buffer.window_start.timestamp() + window_sec) - time.time()
                if 0 < remaining < args.poll_sec:
                    sleep_time = min(sleep_time, max(0.5, remaining))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"  控制迴圈已停止 (Ctrl+C)")
        print(f"  總步數: {step_count}")
        print(f"  最終 SoC: {soc_tracker.get_soc()*100:.1f}%")
        print(f"  日誌: {logger.path}")
        print(f"{'='*60}")

        # 寫入待機指令
        if not args.dry_run:
            try:
                now_ts = datetime.now(TZ_UTC8)
                write_control_file_vendor(
                    args.command_file,
                    {pp: (now_ts, 0.0, 0.0)},
                    global_ts=now_ts,
                    require_empty=False,
                    max_wait_sec=0.1,
                    max_retries=2,
                    situation_code=4,
                )
                print("  已寫入待機指令")
            except Exception:
                pass

    except Exception as e:
        print(f"\n  ✗ 錯誤: {e}")
        import traceback
        traceback.print_exc()

        # 安全寫入待機
        if not args.dry_run:
            try:
                now_ts = datetime.now(TZ_UTC8)
                write_control_file_vendor(
                    args.command_file,
                    {pp: (now_ts, 0.0, 0.0)},
                    global_ts=now_ts,
                    require_empty=False,
                    max_wait_sec=0.1,
                    max_retries=2,
                    situation_code=4,
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()

