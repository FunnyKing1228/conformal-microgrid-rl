"""
P302 真實硬體配置
=================
鋅空氣電池微電網測試台 — 對應 data/raw/ 收集的實測資料

硬體規格：
  ・電池     : 鋅空氣電池 (SLFB), 10 mAh, 充電 8.5V / 放電 5.6V, 額定電流 20 mA
  ・MPPT     : 峰值 ~1.75 W (實測), 典型日間 700-800 mW
  ・電子負載  : 4 組 × 12W @5V (最大 48W)
  ・採樣間隔  : ~11 秒 → 聚合為 15 分鐘窗格

資料來源：
  ・MPPT/Solar → 真實太陽能時間序列（from data/raw/）
  ・負載       → 由我們透過 load_pattern 控制（0-4 組）
  ・電池 SoC   → 模擬計算（CSV 中的電池規格不同，不使用）
  ・SoH        → 未來整合 SoH prediction 模型（用充電段預測）
  ・其餘       → 全由模擬運算

RL 控制輸出（對應 Command.txt 格式 PP,功率(mW),流速,）：
  ・功率 (power_mW)     : 電池充放電功率，0~170 mW
  ・流速 (flow_percent) : 電解液流速 0~100%

Flow Rate 電化學等效模型（SLFB Synthetic Model）：
  ・基線內阻 R_base = (V_charge - V_discharge) / (2 × I_rated) = 72.5 Ω
  ・幫浦寄生功耗 P_pump(Q) = P_max × Q³ （立方律）
  ・等效內阻 R_eq(Q) = R_base × (1 + k_R × (1-Q)/Q)
  ・淨功率 P_net = (V_cell × I) - P_pump(Q)
  ・低流速 → 濃度極化 → 內阻飆升 → 效率下降
  ・高流速 → 反應物充足 → 效率好 → 但幫浦功耗增加
  ・RL Agent 需學會平衡流速：高流速效率好但幫浦耗電，低流速省幫浦但效率差
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
BATTERY_EFFICIENCY      = 0.85           # 鋅空氣電池充放電基礎效率（保守估計）

# ──────────────────────────────────────────────────────────────
# Flow Rate 電化學等效模型參數
# ──────────────────────────────────────────────────────────────
# 基線內阻 R_base = (V_charge - V_discharge) / (2 × I_rated)
FLOW_R_BASE_OHM         = (BATTERY_CHARGE_V - BATTERY_DISCHARGE_V) / (2 * BATTERY_CHARGE_I_MA / 1000)  # 72.5 Ω
# 幫浦最大寄生功率 ≈ 15% 放電功率
FLOW_P_MAX_PUMP_W       = (BATTERY_DISCHARGE_V * BATTERY_CHARGE_I_MA / 1000) * 0.15  # 0.0168 W (16.8 mW)
# 內阻增幅因子（可調超參）
FLOW_K_R                = 0.5
# 開路電壓（充放電基準）
FLOW_V_OCV_CHARGE       = BATTERY_CHARGE_V     # 8.5 V
FLOW_V_OCV_DISCHARGE    = BATTERY_DISCHARGE_V  # 5.6 V
# 額定電流
FLOW_I_RATED_A          = BATTERY_CHARGE_I_MA / 1000  # 0.020 A

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
        battery_efficiency      = BATTERY_EFFICIENCY,
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

        # ── Flow Rate 電化學等效模型 ──
        use_flow_rate_action    = use_extended_obs,  # 擴充模式自動啟用 2D action
        flow_R_base_ohm         = FLOW_R_BASE_OHM,
        flow_P_max_pump_W       = FLOW_P_MAX_PUMP_W,
        flow_k_R                = FLOW_K_R,
        flow_V_OCV_charge       = FLOW_V_OCV_CHARGE,
        flow_V_OCV_discharge    = FLOW_V_OCV_DISCHARGE,
        flow_I_rated_A          = FLOW_I_RATED_A,

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
    print("=" * 60)
    print(f"Battery Capacity  : {BATTERY_CAPACITY_MAH} mAh = {BATTERY_CAPACITY_WH:.4f} Wh = {BATTERY_CAPACITY_KWH:.6f} kWh")
    print(f"Battery Power     : {BATTERY_POWER_W:.4f} W = {BATTERY_POWER_KW:.6f} kW")
    print(f"Battery Efficiency: {BATTERY_EFFICIENCY}")
    print(f"Charge V / I      : {BATTERY_CHARGE_V} V / {BATTERY_CHARGE_I_MA} mA")
    print(f"Discharge V       : {BATTERY_DISCHARGE_V} V")
    print(f"MPPT Peak         : {MPPT_PEAK_W} W = {MPPT_PEAK_KW:.6f} kW")
    print(f"Load Max          : {LOAD_MAX_W} W = {LOAD_MAX_KW:.4f} kW ({LOAD_GROUPS} groups)")
    print(f"Time Step         : {TIME_STEP_H} h = {AGGREGATION_WINDOW_MIN} min")
    print(f"Steps/Day         : {STEPS_PER_DAY}")
    print(f"Training CSV      : {TRAINING_CSV}")
    print(f"\nFull charge time from 0%: {BATTERY_CAPACITY_MAH/BATTERY_CHARGE_I_MA*60:.1f} min")
    print(f"Energy per charge cycle : {BATTERY_CAPACITY_WH:.4f} Wh")

    print(f"\n{'='*60}")
    print("Flow Rate Model (SLFB Synthetic)")
    print(f"{'='*60}")
    print(f"R_base            : {FLOW_R_BASE_OHM:.1f} Ohm")
    print(f"P_max_pump        : {FLOW_P_MAX_PUMP_W*1000:.1f} mW ({FLOW_P_MAX_PUMP_W:.4f} W)")
    print(f"k_R               : {FLOW_K_R}")
    print(f"V_OCV charge      : {FLOW_V_OCV_CHARGE} V")
    print(f"V_OCV discharge   : {FLOW_V_OCV_DISCHARGE} V")
    print(f"I_rated           : {FLOW_I_RATED_A*1000:.0f} mA")

    # 顯示不同流速下的效能
    print(f"\n{'Q%':>5} | {'R_eq(Ohm)':>10} | {'V_dis(V)':>8} | {'V_chg(V)':>8} | {'P_pump(mW)':>10} | {'eta_dis':>7} | {'eta_chg':>7}")
    print("-" * 75)
    I = FLOW_I_RATED_A
    for q_pct in [1, 5, 10, 25, 50, 75, 100]:
        Q = q_pct / 100.0
        R_eq = FLOW_R_BASE_OHM * (1.0 + FLOW_K_R * (1.0 - Q) / Q)
        V_dis = max(0, FLOW_V_OCV_DISCHARGE - I * R_eq)
        V_chg = FLOW_V_OCV_CHARGE + I * R_eq
        P_pump = FLOW_P_MAX_PUMP_W * Q**3 * 1000  # mW
        eta_dis = V_dis / max(FLOW_V_OCV_DISCHARGE, 1e-6)
        eta_chg = FLOW_V_OCV_CHARGE / max(V_chg, 1e-6)
        print(f"{q_pct:>4}% | {R_eq:>10.1f} | {V_dis:>8.3f} | {V_chg:>8.3f} | {P_pump:>10.3f} | {eta_dis:>7.3f} | {eta_chg:>7.3f}")
