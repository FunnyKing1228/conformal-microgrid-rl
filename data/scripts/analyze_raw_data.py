"""
analyze_raw_data.py
===================
對 data/raw/*.csv 進行完整的 EDA（探索性資料分析）。
重點：mppt_p (MPPT 功率)、時間缺口處理、SoC、溫度。

輸出：
  data/processed/analysis/  資料夾
    ├── data_quality_report.csv   各檔品質報告
    ├── daily_mppt_stats.csv      每日 MPPT 統計
    ├── gap_report.csv            時間缺口明細
    ├── merged_15min.csv          聚合後的 15 分鐘資料（可直接餵模型）
    └── figures/
        ├── mppt_timeseries.png   MPPT 功率時間序列（所有日期）
        ├── mppt_daily_profile.png  平均每日曲線（每小時）
        ├── soc_timeseries.png    SoC 趨勢
        ├── gap_distribution.png  缺口分佈直方圖
        └── correlation_heatmap.png 欄位相關矩陣
"""

import os
import sys
import glob
import warnings
# Windows cp950 終端相容：強制 UTF-8 輸出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')

# ── 路徑設定 ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT  = SCRIPT_DIR.parent
RAW_DIR    = DATA_ROOT / 'raw'
OUT_DIR    = DATA_ROOT / 'processed' / 'analysis'
FIG_DIR    = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_MIN = 15          # 聚合窗格（分鐘）
GAP_THRESH_SEC = 60      # 超過此秒數視為時間缺口
MPPT_MAX_W = 500.0       # 合理 MPPT 功率上限（W），用於異常值過濾


# ══════════════════════════════════════════════════════════════════
# 0. 欄位正規化：統一兩種 CSV 格式 → 共同欄位（SI 單位）
#    格式 A（Jan 5-21）: mppt_p[W], current_a[A], solar_i[A], solar_p[W]
#    格式 B（Jan 22+ ）: mppt_p_mw[mW], current_ma[mA], solar_i_ma[mA], solar_p_mw[mW]
# ══════════════════════════════════════════════════════════════════
def normalize_columns(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    df = df.copy()
    fmt = 'A'  # 預設格式 A

    # 偵測格式
    if 'mppt_p_mw' in df.columns:
        fmt = 'B'

    if fmt == 'B':
        # 功率：mW → W
        for old, new in [('mppt_p_mw', 'mppt_p'), ('solar_p_mw', 'solar_p')]:
            if old in df.columns:
                df[new] = df[old].astype(float) / 1000.0
        # 電流：mA → A
        for old, new in [('current_ma', 'current_a'), ('solar_i_ma', 'solar_i'),
                         ('mppt_i_ma', 'mppt_i')]:
            if old in df.columns:
                df[new] = df[old].astype(float) / 1000.0
        # 其他欄位直接保留（voltage_v、mppt_v 單位不變）

    # 確保標準欄位都存在（填 NaN 若缺）
    for col in ['mppt_p', 'mppt_v', 'mppt_i', 'solar_p', 'solar_v', 'solar_i',
                'current_a', 'voltage_v', 'soc_percent', 'temp_c', 'speed_percent']:
        if col not in df.columns:
            df[col] = np.nan

    return df, fmt


# ══════════════════════════════════════════════════════════════════
# 1. 載入所有 CSV
# ══════════════════════════════════════════════════════════════════
def load_all_raw(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob('collected_data_*.csv'))
    if not files:
        raise FileNotFoundError(f"找不到 CSV 於 {raw_dir}")
    
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=['timestamp'])
            df, fmt = normalize_columns(df, f.name)
            df['_source_file'] = f.name
            df['_fmt'] = fmt
            frames.append(df)
            print(f"  OK {f.name:45s}  {len(df):>5d} rows  [fmt {fmt}]")
        except Exception as e:
            print(f"  ERR {f.name}  ERROR: {e}")
    
    if not frames:
        raise RuntimeError("無法載入任何 CSV")
    
    # 只保留共同欄位（避免不同格式欄位衝突）
    keep_cols = ['timestamp', 'battery_id', 'soc_percent', 'voltage_v', 'current_a',
                 'temp_c', 'speed_percent', 'solar_v', 'solar_i', 'solar_p',
                 'mppt_v', 'mppt_i', 'mppt_p', '_source_file', '_fmt']
    cleaned = []
    for df in frames:
        existing = [c for c in keep_cols if c in df.columns]
        cleaned.append(df[existing])
    
    df_all = pd.concat(cleaned, ignore_index=True)
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    return df_all


# ══════════════════════════════════════════════════════════════════
# 2. 資料品質檢查
# ══════════════════════════════════════════════════════════════════
def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fname, grp in df.groupby('_source_file'):
        grp = grp.sort_values('timestamp')
        n = len(grp)
        ts = grp['timestamp']
        diffs = ts.diff().dt.total_seconds().dropna()
        dt_median = diffs.median()
        dt_std    = diffs.std()
        n_gaps    = int((diffs > GAP_THRESH_SEC).sum())
        total_gap_min = float(diffs[diffs > GAP_THRESH_SEC].sum() / 60)
        mppt_p    = grp['mppt_p']
        rows.append({
            'file'         : fname,
            'n_rows'       : n,
            'start'        : str(ts.min()),
            'end'          : str(ts.max()),
            'duration_h'   : round((ts.max() - ts.min()).total_seconds() / 3600, 2),
            'dt_median_sec': round(dt_median, 1),
            'dt_std_sec'   : round(dt_std, 1),
            'n_gaps'       : n_gaps,
            'total_gap_min': round(total_gap_min, 1),
            'mppt_p_mean_W': round(mppt_p.mean(), 4),
            'mppt_p_max_W' : round(mppt_p.max(), 4),
            'mppt_p_zero_pct': round((mppt_p == 0).mean() * 100, 1),
            'soc_mean'     : round(grp['soc_percent'].mean(), 2),
            'soc_min'      : round(grp['soc_percent'].min(), 2),
            'soc_max'      : round(grp['soc_percent'].max(), 2),
            'n_null'       : int(grp.isnull().sum().sum()),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# 3. 時間缺口分析
# ══════════════════════════════════════════════════════════════════
def gap_report(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('timestamp').reset_index(drop=True)
    diffs = df['timestamp'].diff().dt.total_seconds()
    gap_mask = diffs > GAP_THRESH_SEC
    gap_idx  = diffs[gap_mask].index

    rows = []
    for i in gap_idx:
        t_before = df.loc[i-1, 'timestamp']
        t_after  = df.loc[i,   'timestamp']
        gap_sec  = float(diffs.loc[i])
        rows.append({
            'gap_start'   : str(t_before),
            'gap_end'     : str(t_after),
            'gap_sec'     : round(gap_sec, 1),
            'gap_min'     : round(gap_sec / 60, 2),
            'source_file' : df.loc[i, '_source_file'],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# 4. 異常值清理
# ══════════════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 負值設為 0（物理不合理）
    for col in ['mppt_p', 'mppt_v', 'mppt_i', 'solar_p', 'solar_v', 'solar_i']:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                print(f"  [{col}] 清除 {n_neg} 筆負值 → 0")
            df[col] = df[col].clip(lower=0.0)
    
    # mppt_p 超過合理上限 → NaN（後續插值）
    n_over = (df['mppt_p'] > MPPT_MAX_W).sum()
    if n_over > 0:
        print(f"  [mppt_p] 標記 {n_over} 筆超限值（>{MPPT_MAX_W}W）為 NaN")
        df.loc[df['mppt_p'] > MPPT_MAX_W, 'mppt_p'] = np.nan
    
    # SoC 範圍檢查
    n_soc = ((df['soc_percent'] < 0) | (df['soc_percent'] > 100)).sum()
    if n_soc > 0:
        print(f"  [soc_percent] 清除 {n_soc} 筆越界值")
        df['soc_percent'] = df['soc_percent'].clip(0.0, 100.0)
    
    # 時間缺口期間的 mppt_p：缺口 > 5 分鐘 → 保持 NaN（不插值）
    #                         缺口 ≤ 5 分鐘 → 線性插值
    diffs = df['timestamp'].diff().dt.total_seconds().fillna(0)
    long_gap_starts = df.index[diffs > 300]  # 5 分鐘
    df = df.set_index('timestamp')
    df['mppt_p'] = df['mppt_p'].interpolate(method='time', limit=30)
    df = df.reset_index()
    # 長缺口之後的第一筆不插值（保持 NaN 或原值）
    for idx in long_gap_starts:
        if idx < len(df):
            # 已無法修改，interpolate 已完成，但記錄供參考
            pass
    
    return df


# ══════════════════════════════════════════════════════════════════
# 5. 15 分鐘聚合
# ══════════════════════════════════════════════════════════════════
def aggregate_15min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index('timestamp').sort_index()
    rule = f'{WINDOW_MIN}T'

    # 取樣間隔（秒）
    diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    dt_sec = float(diffs.median()) if len(diffs) > 0 else 10.0
    dt_h   = dt_sec / 3600.0

    agg = pd.DataFrame()
    agg['mppt_p_mean_W']  = df['mppt_p'].resample(rule).mean()
    agg['mppt_p_std_W']   = df['mppt_p'].resample(rule).std().fillna(0.0)
    agg['mppt_p_max_W']   = df['mppt_p'].resample(rule).max()
    agg['mppt_p_min_W']   = df['mppt_p'].resample(rule).min()
    agg['mppt_v_mean_V']  = df['mppt_v'].resample(rule).mean()
    agg['mppt_i_mean_A']  = df['mppt_i'].resample(rule).mean()
    agg['solar_p_mean_W'] = df['solar_p'].resample(rule).mean()
    agg['soc_mean_pct']   = df['soc_percent'].resample(rule).mean()
    agg['soc_end_pct']    = df['soc_percent'].resample(rule).last()
    agg['voltage_mean_V'] = df['voltage_v'].resample(rule).mean()
    agg['current_mean_A'] = df['current_a'].resample(rule).mean()
    if 'temp_c' in df.columns:
        agg['temp_mean_c'] = df['temp_c'].resample(rule).mean()
    agg['n_samples']      = df['mppt_p'].resample(rule).count()

    # 能量（Wh）
    agg['energy_mppt_wh'] = agg['mppt_p_mean_W'] * agg['n_samples'] * dt_h
    agg['energy_mppt_wh'] = agg['energy_mppt_wh'].fillna(0.0)

    # 資料完整度（實際採樣數 / 理論採樣數）
    expected = WINDOW_MIN * 60 / max(dt_sec, 1.0)
    agg['completeness'] = (agg['n_samples'] / expected).clip(0.0, 1.0).round(3)

    # 標記缺口窗格（完整度 < 50%）
    agg['has_gap'] = (agg['completeness'] < 0.5).astype(int)

    agg = agg.reset_index()
    agg.rename(columns={'timestamp': 'timestamp'}, inplace=True)

    # 時間 context 特徵（for RL）
    agg['hour']        = agg['timestamp'].dt.hour
    agg['minute']      = agg['timestamp'].dt.minute
    agg['day_of_week'] = agg['timestamp'].dt.dayofweek
    agg['date']        = agg['timestamp'].dt.date

    # 別名（相容 MicrogridEnvironment._load_external_csv）
    agg['Solar']       = agg['mppt_p_mean_W'] / 1000.0  # W → kW

    return agg


# ══════════════════════════════════════════════════════════════════
# 6. 每日 MPPT 統計
# ══════════════════════════════════════════════════════════════════
def daily_stats(df15: pd.DataFrame) -> pd.DataFrame:
    grp = df15.groupby('date')
    stats = pd.DataFrame({
        'mppt_p_mean_W'    : grp['mppt_p_mean_W'].mean(),
        'mppt_p_max_W'     : grp['mppt_p_max_W'].max(),
        'energy_mppt_wh'   : grp['energy_mppt_wh'].sum(),
        'soc_start_pct'    : grp['soc_mean_pct'].first(),
        'soc_end_pct'      : grp['soc_end_pct'].last(),
        'n_windows'        : grp['n_samples'].count(),
        'n_gap_windows'    : grp['has_gap'].sum(),
        'mean_completeness': grp['completeness'].mean(),
    }).reset_index()
    stats['energy_mppt_kwh'] = stats['energy_mppt_wh'] / 1000.0
    return stats


# ══════════════════════════════════════════════════════════════════
# 7. 繪圖
# ══════════════════════════════════════════════════════════════════
def plot_mppt_timeseries(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle('MPPT 功率 / SoC / 電流  時間序列（全部日期）', fontsize=14, fontweight='bold')

    # 標記缺口區域（缺口 > 5 分鐘）
    diffs = df['timestamp'].diff().dt.total_seconds().fillna(0)
    gap_starts = df['timestamp'][diffs > 300].values

    ax0, ax1, ax2 = axes

    ax0.plot(df['timestamp'], df['mppt_p'], color='#F4A460', linewidth=0.6, label='mppt_p (W)')
    ax0.set_ylabel('MPPT 功率 (W)', fontsize=10)
    ax0.set_ylim(bottom=-1)
    ax0.legend(loc='upper right', fontsize=8)
    ax0.grid(alpha=0.3)

    ax1.plot(df['timestamp'], df['soc_percent'], color='#4682B4', linewidth=0.6, label='SoC (%)')
    ax1.set_ylabel('SoC (%)', fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(df['timestamp'], df['current_a'], color='#2E8B57', linewidth=0.5, label='電流 (A)')
    ax2.set_ylabel('電流 (A)', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.3)

    # 標記缺口
    for gs in gap_starts:
        for ax in axes:
            ax.axvline(x=gs, color='red', linewidth=0.8, alpha=0.4, linestyle='--')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.tight_layout()
    out = FIG_DIR / 'mppt_timeseries.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  圖表儲存 → {out}")


def plot_daily_profile(df15: pd.DataFrame):
    """平均每日 MPPT 曲線（按小時/15分鐘）"""
    df15 = df15.copy()
    df15['hm'] = df15['hour'] + df15['minute'] / 60.0
    
    # 只取 mppt_p > 0 的窗格（夜間為 0 不影響曲線）
    profile = df15.groupby('hm')['mppt_p_mean_W'].agg(['mean', 'std', 'max']).reset_index()
    profile.columns = ['hour_frac', 'mean', 'std', 'max']

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(profile['hour_frac'],
                    (profile['mean'] - profile['std']).clip(0),
                    profile['mean'] + profile['std'],
                    alpha=0.25, color='#FFA500', label='±1σ')
    ax.plot(profile['hour_frac'], profile['mean'], color='#FF8C00', linewidth=2, label='平均 MPPT (W)')
    ax.plot(profile['hour_frac'], profile['max'],  color='#DC143C',  linewidth=1, linestyle='--', label='峰值 MPPT (W)')
    ax.set_xlabel('時刻（小時）', fontsize=11)
    ax.set_ylabel('MPPT 功率 (W)', fontsize=11)
    ax.set_title('MPPT 平均日曲線（所有記錄日平均）', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / 'mppt_daily_profile.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  圖表儲存 → {out}")


def plot_gap_distribution(gap_df: pd.DataFrame):
    if gap_df.empty:
        print("  [缺口分佈] 無缺口資料，跳過")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('時間缺口分析', fontsize=13, fontweight='bold')

    # 直方圖
    axes[0].hist(gap_df['gap_min'], bins=30, color='#CD5C5C', edgecolor='white')
    axes[0].set_xlabel('缺口長度（分鐘）', fontsize=10)
    axes[0].set_ylabel('次數', fontsize=10)
    axes[0].set_title('缺口長度分佈', fontsize=11)
    axes[0].grid(alpha=0.3)

    # 每個來源檔案的缺口數
    cnt = gap_df.groupby('source_file')['gap_sec'].count().sort_values(ascending=False)
    cnt.plot(kind='barh', ax=axes[1], color='#6495ED')
    axes[1].set_xlabel('缺口次數', fontsize=10)
    axes[1].set_title('各日缺口次數', fontsize=11)
    axes[1].grid(alpha=0.3, axis='x')

    plt.tight_layout()
    out = FIG_DIR / 'gap_distribution.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  圖表儲存 → {out}")


def plot_correlation(df: pd.DataFrame):
    cols = ['mppt_p', 'mppt_v', 'mppt_i', 'solar_p', 'soc_percent', 'voltage_v', 'current_a']
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].dropna().sample(min(5000, len(df)), random_state=42)
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, square=True, linewidths=0.5, annot_kws={'size': 9})
    ax.set_title('欄位相關矩陣', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIG_DIR / 'correlation_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  圖表儲存 → {out}")


def plot_daily_energy(daily_df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    dates = [str(d) for d in daily_df['date']]
    x = range(len(dates))

    bars = ax1.bar(x, daily_df['energy_mppt_kwh'], color='#DAA520', alpha=0.8, label='日發電量 (kWh)')
    ax1.set_ylabel('日發電量 (kWh)', fontsize=11)
    ax1.set_xlabel('日期', fontsize=11)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(dates, rotation=45, ha='right', fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(list(x), daily_df['mean_completeness'] * 100, 'o-', color='#4169E1',
             linewidth=1.5, markersize=4, label='資料完整度 (%)')
    ax2.set_ylabel('資料完整度 (%)', fontsize=11)
    ax2.set_ylim(0, 110)

    # 合併圖例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize=9)
    ax1.set_title('每日 MPPT 發電量 & 資料完整度', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    out = FIG_DIR / 'daily_energy.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  圖表儲存 → {out}")


# ══════════════════════════════════════════════════════════════════
# 8. 主流程
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  微電網原始資料 EDA")
    print("=" * 60)

    # 1. 載入
    print("\n[1/7] 載入 CSV...")
    df = load_all_raw(RAW_DIR)
    print(f"\n  合計 {len(df):,} 筆，時間範圍：{df['timestamp'].min()}  ~  {df['timestamp'].max()}")

    # 2. 品質報告
    print("\n[2/7] 品質報告...")
    qr = quality_report(df)
    qr.to_csv(OUT_DIR / 'data_quality_report.csv', index=False, encoding='utf-8-sig')
    print(qr[['file','n_rows','duration_h','n_gaps','total_gap_min',
               'mppt_p_mean_W','mppt_p_max_W','mppt_p_zero_pct']].to_string(index=False))

    # 3. 缺口報告
    print("\n[3/7] 缺口分析...")
    gaps = gap_report(df)
    gaps.to_csv(OUT_DIR / 'gap_report.csv', index=False, encoding='utf-8-sig')
    print(f"  共 {len(gaps)} 個缺口，最大 {gaps['gap_min'].max():.1f} 分鐘" if not gaps.empty else "  無缺口")

    # 4. 清理
    print("\n[4/7] 資料清理（負值、異常值、短缺口插值）...")
    df_clean = clean_data(df)

    # 5. 15 分鐘聚合
    print("\n[5/7] 15 分鐘聚合...")
    df15 = aggregate_15min(df_clean)
    df15.to_csv(OUT_DIR.parent / 'merged_15min.csv', index=False, encoding='utf-8-sig')
    print(f"  輸出 {len(df15)} 個 15-min 窗格 → data/processed/merged_15min.csv")

    # 6. 每日統計
    print("\n[6/7] 每日統計...")
    daily = daily_stats(df15)
    daily.to_csv(OUT_DIR / 'daily_mppt_stats.csv', index=False, encoding='utf-8-sig')
    print(daily[['date','energy_mppt_kwh','mppt_p_max_W','n_gap_windows','mean_completeness']].to_string(index=False))

    # 7. 繪圖
    print("\n[7/7] 繪圖...")
    plot_mppt_timeseries(df_clean)
    plot_daily_profile(df15)
    plot_gap_distribution(gaps)
    plot_correlation(df_clean)
    plot_daily_energy(daily)

    # ── 總結 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  分析完成！輸出目錄：")
    print(f"  {OUT_DIR}")
    print("=" * 60)
    print("\n  關鍵統計：")
    total_wh = daily['energy_mppt_wh'].sum()
    print(f"  ・總 MPPT 發電量    : {total_wh/1000:.3f} kWh（{len(daily)} 天）")
    print(f"  ・每日平均發電量    : {total_wh/len(daily)/1000:.3f} kWh/day")
    print(f"  ・MPPT 峰值功率     : {df_clean['mppt_p'].max():.2f} W")
    print(f"  ・MPPT 平均非零功率 : {df_clean.loc[df_clean['mppt_p']>0,'mppt_p'].mean():.2f} W")
    print(f"  ・資料完整度（平均）: {df15['completeness'].mean()*100:.1f}%")
    print(f"  ・總缺口次數        : {len(gaps)}")
    print(f"  ・總缺口時間        : {gaps['gap_min'].sum():.1f} 分鐘" if not gaps.empty else "  ・無缺口")

    print("\n  RL 建議設定：")
    p_scale = df_clean['mppt_p'].quantile(0.95)
    print(f"  ・synthetic_pv_peak_kw ≈ {p_scale/1000:.3f} kW（95th percentile）")
    dt_sec = df_clean['timestamp'].diff().dt.total_seconds().median()
    print(f"  ・採樣間隔中位數    : {dt_sec:.1f} 秒")
    print(f"  ・建議 time_step    : 0.25（小時，= 15 分鐘聚合）")


if __name__ == '__main__':
    main()
