"""
build_training_dataset.py
=========================
從 data/raw/ 選取最佳連續天的 MPPT/Solar 資料，加上：
  1. 負載模式注入（4 組 × 12W @5V，由我們控制）
  2. 15 分鐘聚合

重要：
  ・CSV 中只取 MPPT（太陽能）數據 — 這是真實的可再生能源時間序列
  ・CSV 中的電池欄位（soc_percent, voltage_v, current_a）為不同規格的電池，不使用
  ・SoC / SoH / Flow Rate 全由模擬計算，不依賴 CSV 電池數據
  ・負載由我們透過 load_pattern 控制（0-4 組 × 12W）

電池規格（目標鋅空氣電池，模擬用）：
  - 額定容量      : 10 mAh
  - 充電電流      : 20 mA
  - 充電電壓      : 8.5 V
  - 放電電壓      : 5.6 V

RL 控制輸出：
  - 功率 (power_mW)     : 充放電功率
  - 流速 (flow_percent) : 電解液流速（0-100%）

輸出 → data/processed/training_7day_15min.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, time as dtime

# Windows 相容
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT  = SCRIPT_DIR.parent
RAW_DIR    = DATA_ROOT / 'raw'
OUT_DIR    = DATA_ROOT / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 電池規格
# ═══════════════════════════════════════════════════════════════
BATTERY_CAPACITY_MAH  = 10.0       # mAh
BATTERY_CHARGE_V      = 8.5        # V
BATTERY_DISCHARGE_V   = 5.6        # V
BATTERY_CHARGE_I_MA   = 20.0       # mA（額定充電電流）
BATTERY_CAPACITY_WH   = BATTERY_CAPACITY_MAH * (BATTERY_CHARGE_V + BATTERY_DISCHARGE_V) / 2 / 1000  # ≈ 0.0705 Wh
BATTERY_CAPACITY_KWH  = BATTERY_CAPACITY_WH / 1000

# ═══════════════════════════════════════════════════════════════
# 負載規格
# ═══════════════════════════════════════════════════════════════
LOAD_PER_GROUP_W = 12.0   # W
LOAD_VOLTAGE     = 5.0    # V
MAX_GROUPS       = 4

# 負載時刻表（from load_pattern.txt，已修正為 4 組）
LOAD_SCHEDULE = [
    (dtime( 0, 0), 0),  (dtime( 1, 0), 0),  (dtime( 2, 0), 0),
    (dtime( 3, 0), 0),  (dtime( 4, 0), 0),  (dtime( 5, 0), 0),
    (dtime( 6, 0), 1),  (dtime( 7, 0), 1),  (dtime( 8, 0), 2),
    (dtime( 9, 0), 3),  (dtime(10, 0), 4),  (dtime(11, 0), 4),
    (dtime(12, 0), 4),  (dtime(13, 0), 4),  (dtime(14, 0), 4),
    (dtime(15, 0), 3),  (dtime(16, 0), 3),  (dtime(17, 0), 2),
    (dtime(18, 0), 2),  (dtime(19, 0), 1),  (dtime(20, 0), 1),
    (dtime(21, 0), 0),  (dtime(22, 0), 0),  (dtime(23, 0), 0),
]


def get_load_groups(t: dtime) -> int:
    """根據時間查表回傳負載組數"""
    result = 0
    for sched_t, n in LOAD_SCHEDULE:
        if t >= sched_t:
            result = n
    return result


# ═══════════════════════════════════════════════════════════════
# 欄位正規化（與 analyze_raw_data.py 相同邏輯）
# ═══════════════════════════════════════════════════════════════
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'mppt_p_mw' in df.columns:
        for old, new in [('mppt_p_mw', 'mppt_p'), ('solar_p_mw', 'solar_p')]:
            if old in df.columns:
                df[new] = df[old].astype(float) / 1000.0
        for old, new in [('current_ma', 'current_a'), ('solar_i_ma', 'solar_i'),
                         ('mppt_i_ma', 'mppt_i')]:
            if old in df.columns:
                df[new] = df[old].astype(float) / 1000.0
    for col in ['mppt_p', 'mppt_v', 'mppt_i', 'solar_p', 'solar_v', 'solar_i',
                'current_a', 'voltage_v', 'soc_percent', 'temp_c']:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ═══════════════════════════════════════════════════════════════
# 載入 & 選取 7 天
# ═══════════════════════════════════════════════════════════════
def load_and_select(raw_dir: Path) -> pd.DataFrame:
    """
    載入 raw CSV，只看 MPPT/Solar 品質選日期（忽略電池欄位）。

    選取條件：
      1. 完整度 ≥ 90%（約 24h 連續紀錄）
      2. 白天 MPPT 有明確訊號（mppt_p 非全零）
      3. 取最佳連續天
    """
    files = sorted(raw_dir.glob('collected_data_*.csv'))
    daily = {}
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=['timestamp'])
            df = normalize_columns(df)
            date = df['timestamp'].dt.date.iloc[0]
            diffs = df['timestamp'].diff().dt.total_seconds().dropna()
            dt_med = diffs.median() if len(diffs) > 0 else 11.0
            expected = 24 * 3600 / dt_med
            completeness = min(len(df) / expected, 1.0)
            dur_h = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            # 白天 MPPT 品質
            mask_d = (df['timestamp'].dt.hour >= 8) & (df['timestamp'].dt.hour <= 16)
            mppt_day_mean = df.loc[mask_d, 'mppt_p'].mean() if mask_d.sum() > 0 else 0.0
            mppt_day_max  = df.loc[mask_d, 'mppt_p'].max()  if mask_d.sum() > 0 else 0.0
            daily[date] = {
                'file': f, 'df': df, 'n_rows': len(df),
                'completeness': completeness, 'duration_h': dur_h,
                'mppt_day_mean_W': mppt_day_mean,
                'mppt_day_max_W': mppt_day_max,
            }
            mppt_ok = 'OK' if mppt_day_mean > 0.01 else 'low'
            print(f"  {f.name:45s}  {len(df):>5d} rows  {dur_h:5.1f}h  "
                  f"完整度 {completeness*100:5.1f}%  MPPT:{mppt_ok}({mppt_day_mean*1000:.0f}mW)")
        except Exception as e:
            print(f"  ERR {f.name}: {e}")

    # 找出所有完整度 >= 90% 且 MPPT 有訊號的日期
    from datetime import date as ddate, timedelta
    good_dates = sorted([d for d, v in daily.items()
                         if v['completeness'] >= 0.90 and v['mppt_day_mean_W'] > 0.01])
    print(f"\n  品質合格日期（完整度≥90% & MPPT>10mW）：{good_dates}")

    # 找最長連續段
    if good_dates:
        best_run = [good_dates[0]]
        current_run = [good_dates[0]]
        for i in range(1, len(good_dates)):
            if (good_dates[i] - good_dates[i-1]).days == 1:
                current_run.append(good_dates[i])
            else:
                if len(current_run) > len(best_run):
                    best_run = current_run
                current_run = [good_dates[i]]
        if len(current_run) > len(best_run):
            best_run = current_run
        selected = best_run
    else:
        # fallback: 取完整度最高的 6 天
        ranked = sorted(daily.items(), key=lambda x: x[1]['completeness'], reverse=True)
        selected = sorted([d for d, _ in ranked[:6]])

    print(f"\n  選定 {len(selected)} 天（最長連續段）：")
    for d in selected:
        v = daily[d]
        print(f"    {d}  完整度 {v['completeness']*100:5.1f}%  "
              f"MPPT day avg {v['mppt_day_mean_W']*1000:.0f}mW  max {v['mppt_day_max_W']*1000:.0f}mW")

    frames = [daily[d]['df'] for d in selected]
    df_all = pd.concat(frames, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    print(f"  合計 {len(df_all):,} 筆")
    return df_all, selected


# ═══════════════════════════════════════════════════════════════
# SoC 庫倫計數
# ═══════════════════════════════════════════════════════════════
def simulate_soc(df: pd.DataFrame) -> np.ndarray:
    """
    模擬 SoC：基於 MPPT 功率和負載功率的能量平衡。

    邏輯（簡化模擬，非 CSV 電池數據）：
      - 充電功率 = min(mppt_p, 額定充電功率 0.17W)
      - 放電功率 = 0（目前無真實放電控制，由 RL agent 決定）
      - 淨充電能量 = (P_charge × dt) × efficiency / capacity

    注意：真實 SoC 計算會由 RL env 內部處理。
    此處只提供「被動充電」的 SoC 參考序列作為 dataset 欄位。
    """
    cap_wh = BATTERY_CAPACITY_WH  # ≈ 0.0705 Wh
    max_charge_w = BATTERY_CHARGE_I_MA * BATTERY_CHARGE_V / 1000  # 0.17 W
    efficiency = 0.85

    soc = np.zeros(len(df), dtype=float)
    soc[0] = 0.5  # 初始 50%

    timestamps = df['timestamp'].values
    mppt_p = df['mppt_p'].fillna(0.0).values  # MPPT 功率 (W)

    for i in range(1, len(df)):
        dt_sec = (timestamps[i] - timestamps[i-1]) / np.timedelta64(1, 's')
        if dt_sec <= 0 or dt_sec > 600:
            soc[i] = soc[i-1]
            continue

        dt_h = dt_sec / 3600.0

        # 充電功率 = MPPT 輸出，限於額定充電功率
        p_charge_w = min(float(mppt_p[i]), max_charge_w)
        p_charge_w = max(p_charge_w, 0.0)

        # 能量平衡（純充電，放電由 RL 決定）
        delta_wh = p_charge_w * dt_h * efficiency
        delta_soc = delta_wh / cap_wh if cap_wh > 0 else 0.0

        soc[i] = float(np.clip(soc[i-1] + delta_soc, 0.0, 1.0))

    return soc


# ═══════════════════════════════════════════════════════════════
# 15 分鐘聚合
# ═══════════════════════════════════════════════════════════════
WINDOW_MIN = 15

def aggregate_15min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().set_index('timestamp').sort_index()
    rule = f'{WINDOW_MIN}min'

    diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    dt_sec = float(diffs.median()) if len(diffs) > 0 else 11.0
    dt_h   = dt_sec / 3600.0

    agg = pd.DataFrame()

    # MPPT（W）
    agg['mppt_p_mean_W']  = df['mppt_p'].resample(rule).mean()
    agg['mppt_p_std_W']   = df['mppt_p'].resample(rule).std().fillna(0.0)
    agg['mppt_p_max_W']   = df['mppt_p'].resample(rule).max()
    agg['mppt_v_mean_V']  = df['mppt_v'].resample(rule).mean()
    agg['mppt_i_mean_A']  = df['mppt_i'].resample(rule).mean()

    # Solar
    agg['solar_p_mean_W'] = df['solar_p'].resample(rule).mean()

    # 溫度（若有值才取）
    if 'temp_c' in df.columns and df['temp_c'].notna().any():
        agg['temp_mean_c'] = df['temp_c'].resample(rule).mean()

    # SoC（庫倫計數估算）
    agg['soc_mean']       = df['soc_estimated'].resample(rule).mean()
    agg['soc_end']        = df['soc_estimated'].resample(rule).last()

    # 負載
    agg['load_groups']    = df['load_groups'].resample(rule).mean()  # 平均組數
    agg['load_W']         = df['load_W'].resample(rule).mean()
    agg['load_std_W']     = df['load_W'].resample(rule).std().fillna(0.0)
    agg['load_max_W']     = df['load_W'].resample(rule).max()

    # 採樣數 & 完整度
    counts = df['mppt_p'].resample(rule).count()
    agg['n_samples'] = counts
    expected = WINDOW_MIN * 60 / max(dt_sec, 1.0)
    agg['completeness'] = (counts / expected).clip(0.0, 1.0).round(3)
    agg['has_gap'] = (agg['completeness'] < 0.5).astype(int)

    # 能量（Wh）
    agg['energy_mppt_Wh'] = agg['mppt_p_mean_W'] * counts * dt_h
    agg['energy_load_Wh'] = agg['load_W'] * WINDOW_MIN / 60.0  # W × h

    agg = agg.reset_index()

    # 時間 context
    agg['hour']        = agg['timestamp'].dt.hour
    agg['minute']      = agg['timestamp'].dt.minute
    agg['day_of_week'] = agg['timestamp'].dt.dayofweek
    agg['date']        = agg['timestamp'].dt.date

    # 別名（相容 MicrogridEnvironment）
    agg['Solar']       = agg['mppt_p_mean_W'] / 1000.0  # W → kW
    agg['Consumption'] = agg['load_W'] / 1000.0          # W → kW

    # 電價模型（TOU）
    base_price = 0.15
    agg['price'] = np.where(
        (agg['hour'] >= 8) & (agg['hour'] <= 18),
        base_price * 1.2, base_price * 0.8
    )

    return agg


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Training Dataset Builder")
    print("  電池：10 mAh / 20 mA / 8.5V(充) / 5.6V(放)")
    print("  負載：4 組 x 12W @5V")
    print("=" * 60)

    # 1. 載入 & 選 7 天
    print("\n[1/4] 載入資料 & 選取最佳 7 天...")
    df, selected_dates = load_and_select(RAW_DIR)

    # 2. 注入負載
    print("\n[2/4] 注入負載模式...")
    df['load_groups'] = df['timestamp'].apply(
        lambda ts: get_load_groups(ts.time())
    )
    df['load_W'] = df['load_groups'] * LOAD_PER_GROUP_W
    load_summary = df.groupby('load_groups')['load_W'].count()
    print("  負載分佈（採樣數）：")
    for n, cnt in load_summary.items():
        print(f"    {int(n)} 組 ({int(n)*12}W)：{cnt:,} 筆")

    # 3. SoC 模擬（被動充電）
    print("\n[3/4] SoC 模擬（MPPT→電池被動充電）...")
    df['soc_estimated'] = simulate_soc(df)
    soc = df['soc_estimated']
    print(f"  SoC 範圍：{soc.min():.4f} ~ {soc.max():.4f}")
    print(f"  SoC 平均：{soc.mean():.4f}")
    print(f"  SoC 末端：{soc.iloc[-1]:.4f}")

    # 4. 15 分鐘聚合
    print("\n[4/4] 15 分鐘聚合...")
    df_agg = aggregate_15min(df)

    # 移除完整度過低的窗格
    n_before = len(df_agg)
    df_agg = df_agg[df_agg['completeness'] > 0.3].reset_index(drop=True)
    n_after = len(df_agg)
    print(f"  聚合窗格：{n_before} → {n_after}（移除 {n_before - n_after} 個低品質窗格）")

    # 輸出
    out_path = OUT_DIR / 'training_7day_15min.csv'
    df_agg.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n  輸出 → {out_path}")

    # 也輸出原始逐筆資料供分析（只含 MPPT/Solar + 模擬欄位）
    raw_out = OUT_DIR / 'training_7day_raw.csv'
    keep = ['timestamp', 'mppt_p', 'mppt_v', 'mppt_i', 'solar_p',
            'load_groups', 'load_W', 'soc_estimated']
    df[[c for c in keep if c in df.columns]].to_csv(raw_out, index=False, encoding='utf-8-sig')
    print(f"  原始資料 → {raw_out}")

    # ── 統計摘要 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Dataset 統計摘要")
    print("=" * 60)
    print(f"  選定日期        ：{selected_dates[0]} ~ {selected_dates[-1]}")
    print(f"  原始採樣數      ：{len(df):,}")
    print(f"  15-min 窗格數   ：{len(df_agg)}")
    print(f"  MPPT 發電總量   ：{df_agg['energy_mppt_Wh'].sum():.3f} Wh")
    print(f"  負載耗電總量    ：{df_agg['energy_load_Wh'].sum():.3f} Wh")
    print(f"  MPPT 峰值功率   ：{df_agg['mppt_p_max_W'].max():.3f} W")
    print(f"  平均完整度      ：{df_agg['completeness'].mean()*100:.1f}%")

    print(f"\n  電池參數（供 RL env 使用）：")
    print(f"    battery_capacity_kwh  = {BATTERY_CAPACITY_KWH:.6f}")
    print(f"    battery_power_kw      = {BATTERY_CHARGE_I_MA * BATTERY_CHARGE_V / 1e6:.6f}  (充電功率)")
    print(f"    time_step             = 0.25  (15 分鐘)")

    # 日別統計
    print("\n  日別統計：")
    for d, grp in df_agg.groupby('date'):
        mppt_wh = grp['energy_mppt_Wh'].sum()
        load_wh = grp['energy_load_Wh'].sum()
        soc_s   = grp['soc_mean'].iloc[0] if len(grp) > 0 else 0
        soc_e   = grp['soc_end'].iloc[-1] if len(grp) > 0 else 0
        print(f"    {d}  MPPT {mppt_wh:6.3f} Wh  Load {load_wh:6.1f} Wh  "
              f"SoC {soc_s:.3f}→{soc_e:.3f}  windows {len(grp)}")

    # 繪圖
    print("\n  繪圖...")
    plot_training_summary(df, df_agg)


def plot_training_summary(df_raw: pd.DataFrame, df_agg: pd.DataFrame):
    """產生 training dataset 摘要圖表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig_dir = OUT_DIR / 'analysis' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. SoC + MPPT + Load 時間序列 ──
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    dates = df_raw.index if isinstance(df_raw.index, pd.DatetimeIndex) else pd.to_datetime(df_raw['timestamp'])
    fig.suptitle('Training Dataset — MPPT (real) + Load (controlled) + SoC (simulated)',
                 fontsize=14, fontweight='bold')

    ax0, ax1, ax2 = axes

    # MPPT（真實太陽能資料）
    ax0.plot(df_raw['timestamp'], df_raw['mppt_p'] * 1000, color='#F4A460', linewidth=0.5)
    ax0.set_ylabel('MPPT (mW)')
    ax0.grid(alpha=0.3)
    ax0.set_title('MPPT Power (real solar data)')

    # 負載模式（由我們控制）
    ax1.step(df_raw['timestamp'], df_raw['load_groups'], color='#DC143C', linewidth=0.8, where='post')
    ax1.set_ylabel('Load Groups')
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.grid(alpha=0.3)
    ax1.set_title('Load Pattern (controlled, 0-4 groups × 12W)')

    # SoC（被動充電模擬）
    ax2.plot(df_raw['timestamp'], df_raw['soc_estimated'] * 100, color='#4682B4', linewidth=0.8)
    ax2.set_ylabel('SoC (%)')
    ax2.set_ylim(-5, 105)
    ax2.grid(alpha=0.3)
    ax2.set_title('SoC (simulated passive charging, 10 mAh battery)')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.tight_layout()
    out = fig_dir / 'training_overview.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  training_overview.png")

    # ── 2. SoC 每日循環 ──
    fig, ax = plt.subplots(figsize=(12, 5))
    for d, grp in df_raw.groupby(df_raw['timestamp'].dt.date):
        hours = (grp['timestamp'] - grp['timestamp'].dt.normalize()).dt.total_seconds() / 3600
        ax.plot(hours, grp['soc_estimated'] * 100, linewidth=0.8, label=str(d), alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('SoC (%)')
    ax.set_title('SoC Daily Cycles (Coulomb Counting)', fontweight='bold')
    ax.set_xlim(0, 24)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = fig_dir / 'soc_daily_cycles.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  soc_daily_cycles.png")

    # ── 3. 15 分鐘聚合 MPPT vs Load ──
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(df_agg['timestamp'], df_agg['mppt_p_mean_W'] * 1000, width=0.008,
            color='#DAA520', alpha=0.8, label='MPPT (mW)')
    ax1.set_ylabel('MPPT Power (mW)')
    ax2 = ax1.twinx()
    ax2.step(df_agg['timestamp'], df_agg['load_groups'], color='#DC143C',
             linewidth=1.2, where='post', label='Load Groups', alpha=0.7)
    ax2.set_ylabel('Load Groups')
    ax2.set_ylim(-0.5, 5)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left')
    ax1.set_title('15-min Aggregated: MPPT vs Load', fontweight='bold')
    ax1.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    out = fig_dir / 'mppt_vs_load_15min.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  mppt_vs_load_15min.png")


if __name__ == '__main__':
    main()
