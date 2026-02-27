"""
preprocess_raw_to_15min.py
==========================
將 10~12 秒採樣的原始資料聚合成 15 分鐘統計特徵，
輸出格式可直接餵入 MicrogridEnvironment（dataset_csv_path）。

預期輸入欄位（data/raw/*.csv）：
  timestamp        : ISO 時間字串或 Unix 秒
  solar_p_kw       : 太陽能輸出功率 (kW)
  load_p_kw        : 負載功率 (kW)  [若無則設 dataset_load_kw]
  battery_soc      : 電池 SoC (0~1)  [可選]
  battery_soh      : 電池 SoH (0~1)  [可選，缺少時填 1.0]
  flow_rate_lpm    : 冷卻液流量 L/min [可選，缺少時填 0.0]

輸出（data/processed/YYYY-MM-DD_15min.csv）：
  timestamp        : 每個 15 分鐘窗格的起始時間
  pv_mean          : 平均 PV 功率 (kW)
  pv_std           : PV 波動標準差 (kW)
  pv_max           : 窗格內最大 PV (kW)
  load_mean        : 平均負載功率 (kW)
  load_std         : 負載波動標準差 (kW)
  load_max         : 窗格內最大負載 (kW)
  soc_mean         : 平均 SoC
  soc_end          : 窗格末尾 SoC（最後一個有效值）
  soh_mean         : 平均 SoH（缺少則 1.0）
  flow_rate_mean   : 平均流量 L/min（缺少則 0.0）
  energy_pv_kwh    : 窗格內 PV 發電量 (kWh)
  energy_load_kwh  : 窗格內負載耗電量 (kWh)

用法：
  python data/scripts/preprocess_raw_to_15min.py \
      --input  data/raw/solar_20250101.csv \
      --output data/processed/solar_20250101_15min.csv \
      [--window_min 15] \
      [--load_kw 5.0]        # 若 CSV 無 load_p_kw 欄位，以此固定值填補
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


WINDOW_MIN = 15  # 聚合窗格（分鐘）


def load_raw_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 時間欄位容錯
    for col in ['timestamp', 'time', 'datetime', 'Time', 'Timestamp']:
        if col in df.columns:
            df['timestamp'] = pd.to_datetime(df[col], errors='coerce', utc=False)
            break
    else:
        raise ValueError("找不到時間欄位（timestamp / time / datetime）")

    # PV 欄位容錯（優先 MPPT，比 raw solar 穩定 ~67 倍）
    # 命名慣例：沒寫 m → 基本單位(V/W/kW)，有寫 m(_mw/_ma) → milli
    # 轉 kW：mW × 1e-6, W × 1e-3, kW × 1
    _pv_candidates = [
        ('mppt_p_kw', 1.0),   # MPPT 優先（kW）
        ('mppt_p_mw', 1e-6),  # MPPT (mW → kW)
        ('solar_p_kw', 1.0),  # fallback: raw solar (kW)
        ('pv_kw', 1.0),       # 通用欄名
        ('Solar', 1.0),       # 已處理的別名
        ('solar_p_mw', 1e-6), # raw solar (mW → kW)
    ]
    for col, scale in _pv_candidates:
        if col in df.columns:
            df['pv_kw'] = df[col].astype(float).fillna(0.0) * scale
            break
    else:
        print("[WARN] 找不到 PV 欄位，以 0 填補")
        df['pv_kw'] = 0.0

    # 負載欄位容錯
    for col in ['load_p_kw', 'load_kw', 'Consumption', 'load']:
        if col in df.columns:
            df['load_kw'] = df[col].astype(float).fillna(0.0)
            break
    else:
        df['load_kw'] = None  # 後面由 --load_kw 補填

    # 可選欄位
    df['soc'] = df['battery_soc'].astype(float) if 'battery_soc' in df.columns else np.nan
    df['soh'] = df['battery_soh'].astype(float) if 'battery_soh' in df.columns else 1.0
    df['flow_lpm'] = df['flow_rate_lpm'].astype(float) if 'flow_rate_lpm' in df.columns else 0.0

    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def aggregate_to_window(df: pd.DataFrame, window_min: int, fixed_load_kw: float = None) -> pd.DataFrame:
    """以 window_min 為窗格對 df 做聚合。"""
    df = df.set_index('timestamp')
    rule = f'{window_min}T'

    # 若無負載資料，用固定值填補
    if df['load_kw'].isna().all():
        if fixed_load_kw is None:
            raise ValueError("CSV 無負載欄位，請加 --load_kw <kW>")
        df['load_kw'] = float(fixed_load_kw)
    elif fixed_load_kw is not None:
        # 明確指定時，覆蓋
        df['load_kw'] = float(fixed_load_kw)

    # 推算取樣間隔（秒），用於算 kWh
    diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    dt_sec = float(diffs.median()) if len(diffs) > 0 else 10.0
    dt_h = dt_sec / 3600.0

    agg = pd.DataFrame()
    agg['pv_mean']        = df['pv_kw'].resample(rule).mean()
    agg['pv_std']         = df['pv_kw'].resample(rule).std().fillna(0.0)
    agg['pv_max']         = df['pv_kw'].resample(rule).max()
    agg['load_mean']      = df['load_kw'].resample(rule).mean()
    agg['load_std']       = df['load_kw'].resample(rule).std().fillna(0.0)
    agg['load_max']       = df['load_kw'].resample(rule).max()
    agg['soc_mean']       = df['soc'].resample(rule).mean()
    agg['soc_end']        = df['soc'].resample(rule).last()
    agg['soh_mean']       = df['soh'].resample(rule).mean().fillna(1.0)
    agg['flow_rate_mean'] = df['flow_lpm'].resample(rule).mean().fillna(0.0)

    # kWh = mean_kW × (N_samples × dt_h)
    counts = df['pv_kw'].resample(rule).count()
    agg['energy_pv_kwh']   = agg['pv_mean'] * counts * dt_h
    agg['energy_load_kwh'] = agg['load_mean'] * counts * dt_h

    agg = agg.reset_index()
    agg.rename(columns={'timestamp': 'timestamp'}, inplace=True)

    # 衍生 context 特徵（供環境使用）
    agg['hour']       = agg['timestamp'].dt.hour
    agg['day_of_week'] = agg['timestamp'].dt.dayofweek  # 0=Mon, 6=Sun
    agg['minute']     = agg['timestamp'].dt.minute

    # 簡易電價模型（TOU：8-18h 峰值）
    base_price = 0.15
    agg['price'] = np.where(
        (agg['hour'] >= 8) & (agg['hour'] <= 18),
        base_price * 1.2,
        base_price * 0.8
    )

    # 重新命名讓 MicrogridEnvironment._load_external_csv 可直接讀
    # 別名：Solar=pv_mean, Consumption=load_mean
    agg['Solar']       = agg['pv_mean']
    agg['Consumption'] = agg['load_mean']

    return agg


def main():
    parser = argparse.ArgumentParser(description="Raw 10-12s → 15min 統計特徵聚合")
    parser.add_argument('--input',      required=True,  help='原始 CSV 路徑')
    parser.add_argument('--output',     default=None,   help='輸出 CSV 路徑（預設自動命名到 data/processed/）')
    parser.add_argument('--window_min', type=int, default=WINDOW_MIN, help='聚合窗格（分鐘，預設 15）')
    parser.add_argument('--load_kw',    type=float, default=None, help='固定負載功率（kW），CSV 無 load 欄位時使用')
    args = parser.parse_args()

    # 輸出路徑
    if args.output is None:
        base = Path(args.input).stem
        out_dir = Path(args.input).parent.parent / 'processed'
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f'{base}_{args.window_min}min.csv')

    print(f"[preprocess] Input : {args.input}")
    print(f"[preprocess] Output: {args.output}")
    print(f"[preprocess] Window: {args.window_min} min")

    df_raw = load_raw_csv(args.input)
    print(f"[preprocess] Loaded {len(df_raw)} raw rows, "
          f"time range: {df_raw['timestamp'].min()} ~ {df_raw['timestamp'].max()}")

    df_out = aggregate_to_window(df_raw, args.window_min, fixed_load_kw=args.load_kw)
    df_out.to_csv(args.output, index=False)

    print(f"[preprocess] Done! {len(df_out)} rows → {args.output}")
    print(f"\n  PV mean range  : {df_out['pv_mean'].min():.2f} ~ {df_out['pv_mean'].max():.2f} kW")
    print(f"  Load mean range: {df_out['load_mean'].min():.2f} ~ {df_out['load_mean'].max():.2f} kW")
    if df_out['soc_mean'].notna().any():
        print(f"  SoC range      : {df_out['soc_mean'].min():.3f} ~ {df_out['soc_mean'].max():.3f}")
    print(f"  SoH mean       : {df_out['soh_mean'].mean():.4f}")


if __name__ == '__main__':
    main()
