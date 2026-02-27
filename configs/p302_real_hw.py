"""
P302 真實硬體配置
=================
鋅空氣電池微電網測試台 — 對應 data/raw/ 收集的實測資料

硬體規格：
  ・電池     : 鋅空氣電池, 10 mAh, 充電 8.5V / 放電 5.6V, 額定電流 20 mA
  ・MPPT     : 峰值 ~1.75 W (實測), 典型日間 700-800 mW
  ・電子負載  : 4 組 × 12W @5V (最大 48W)
  ・採樣間隔  : ~11 秒 → 聚合為 15 分鐘窗格

注意：
  ・負載(48W)遠大於電池容量(0.07Wh)和MPPT(~1W)，
    負載不直接由電池供電，而是獨立供電路徑。
  ・RL agent 控制電池充放電，負載為需求context。
"""

import os

# ──────────────────────────────────────────────────────────────
# 電池參數
# ──────────────────────────────────────────────────────────────
BATTERY_CAPACITY_MAH    = 10.0           # mAh
BATTERY_CHARGE_V        = 8.5            # V（充電電壓）
BATTERY_DISCHARGE_V     = 5.6            # V（放電電壓）
BATTERY_AVG_V           = (BATTERY_CHARGE_V + BATTERY_DISCHARGE_V) / 2  # 7.05 V
BATTERY_CHARGE_I_MA     = 20.0           # mA（額定充電電流）
BATTERY_CAPACITY_WH     = BATTERY_CAPACITY_MAH * BATTERY_AVG_V / 1000   # ≈ 0.0705 Wh
BATTERY_CAPACITY_KWH    = BATTERY_CAPACITY_WH / 1000                     # ≈ 7.05e-5 kWh
BATTERY_POWER_W         = BATTERY_CHARGE_I_MA * BATTERY_CHARGE_V / 1000  # ≈ 0.17 W
BATTERY_POWER_KW        = BATTERY_POWER_W / 1000                         # ≈ 1.7e-4 kW

# ──────────────────────────────────────────────────────────────
# 負載參數
# ──────────────────────────────────────────────────────────────
LOAD_GROUPS             = 4              # 組數
LOAD_PER_GROUP_W        = 12.0           # 每組功率 (W)
LOAD_VOLTAGE            = 5.0            # V
LOAD_MAX_W              = LOAD_GROUPS * LOAD_PER_GROUP_W   # 48 W
LOAD_MAX_KW             = LOAD_MAX_W / 1000                # 0.048 kW

# ──────────────────────────────────────────────────────────────
# MPPT/PV 參數（從實測統計）
# ──────────────────────────────────────────────────────────────
MPPT_PEAK_W             = 1.75           # 歷史最大功率 (W)
MPPT_DAY_AVG_W          = 0.75           # 白天平均 (W)
MPPT_PEAK_KW            = MPPT_PEAK_W / 1000
PV_START_HOUR           = 6
PV_END_HOUR             = 18

# ──────────────────────────────────────────────────────────────
# 時間參數
# ──────────────────────────────────────────────────────────────
SAMPLE_INTERVAL_SEC     = 11.0           # 原始採樣間隔
AGGREGATION_WINDOW_MIN  = 15             # 聚合窗格
TIME_STEP_H             = AGGREGATION_WINDOW_MIN / 60.0  # 0.25 h
STEPS_PER_DAY           = int(24 / TIME_STEP_H)          # 96 steps/day
EPISODE_LENGTH_STEPS    = STEPS_PER_DAY                   # 1 天 = 1 episode

# ──────────────────────────────────────────────────────────────
# 電價模型（TOU - Time of Use）
# ──────────────────────────────────────────────────────────────
PRICE_OFF_PEAK          = 0.10           # $/kWh（離峰）
PRICE_ON_PEAK           = 0.18           # $/kWh（尖峰 8-18h）

# ──────────────────────────────────────────────────────────────
# Dataset 路徑
# ──────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
TRAINING_CSV = os.path.join(_PROJECT_ROOT, 'data', 'processed', 'training_7day_15min.csv')


def get_env_kwargs(use_extended_obs: bool = True) -> dict:
    """
    回傳可直接傳入 MicrogridEnvironment(**kwargs) 的字典。

    用法：
        from configs.p302_real_hw import get_env_kwargs
        env = MicrogridEnvironment(**get_env_kwargs())
    """
    kw = dict(
        # 電池
        battery_capacity_kwh    = BATTERY_CAPACITY_KWH,
        battery_power_kw        = BATTERY_POWER_KW,
        battery_efficiency      = 0.85,        # 鋅空氣電池充放電效率（保守估計）
        soc_min                 = 0.05,         # 鋅空氣電池不建議深放
        soc_max                 = 0.95,

        # 時間
        episode_length          = EPISODE_LENGTH_STEPS,
        time_step               = TIME_STEP_H,

        # 合成資料參數（fallback if no CSV）
        synthetic_pv_peak_kw    = MPPT_PEAK_KW,
        synthetic_pv_start_hour = PV_START_HOUR,
        synthetic_pv_end_hour   = PV_END_HOUR,
        synthetic_load_base_kw  = LOAD_PER_GROUP_W / 1000,  # 1 組 = 基準
        synthetic_load_amp_kw   = (LOAD_MAX_W - LOAD_PER_GROUP_W) / 1000,
        synthetic_price_base    = PRICE_OFF_PEAK,
        synthetic_price_peak    = PRICE_ON_PEAK,
        synthetic_price_peak_start = 8,
        synthetic_price_peak_end   = 18,

        # 外部 CSV dataset
        dataset_csv_path        = TRAINING_CSV,
        dataset_pv_column       = 'Solar',           # kW
        dataset_time_column     = 'timestamp',
        dataset_power_scale     = 1.0,                # 已在 CSV 中轉為 kW

        # 擴充狀態空間
        use_extended_obs        = use_extended_obs,
        initial_soh             = 1.0,
        soh_degradation_per_kwh = 0.001,  # 鋅空氣電池衰減較快
        initial_flow_rate_lpm   = 0.0,    # 目前無液冷

        # 欄位對應
        dataset_pv_std_column   = 'mppt_p_std_W',     # 轉 kW 在 env 內做
        dataset_pv_max_column   = 'mppt_p_max_W',
        dataset_load_std_column = 'load_std_W',
        dataset_load_max_column = 'load_max_W',

        # 安全邊界
        ramp_limit_kw           = BATTERY_POWER_KW * 0.5,  # 爬坡限制
        allow_grid_trading      = True,
    )
    return kw


# ──────────────────────────────────────────────────────────────
# 列印摘要
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("P302 Real Hardware Configuration")
    print("=" * 50)
    print(f"Battery Capacity  : {BATTERY_CAPACITY_MAH} mAh = {BATTERY_CAPACITY_WH:.4f} Wh = {BATTERY_CAPACITY_KWH:.6f} kWh")
    print(f"Battery Power     : {BATTERY_POWER_W:.4f} W = {BATTERY_POWER_KW:.6f} kW")
    print(f"Charge V / I      : {BATTERY_CHARGE_V} V / {BATTERY_CHARGE_I_MA} mA")
    print(f"Discharge V       : {BATTERY_DISCHARGE_V} V")
    print(f"MPPT Peak         : {MPPT_PEAK_W} W = {MPPT_PEAK_KW:.6f} kW")
    print(f"Load Max          : {LOAD_MAX_W} W = {LOAD_MAX_KW:.4f} kW ({LOAD_GROUPS} groups)")
    print(f"Time Step         : {TIME_STEP_H} h = {AGGREGATION_WINDOW_MIN} min")
    print(f"Steps/Day         : {STEPS_PER_DAY}")
    print(f"Training CSV      : {TRAINING_CSV}")
    print(f"\nFull charge time from 0%: {BATTERY_CAPACITY_MAH/BATTERY_CHARGE_I_MA*60:.1f} min")
    print(f"Energy per charge cycle : {BATTERY_CAPACITY_WH:.4f} Wh")
