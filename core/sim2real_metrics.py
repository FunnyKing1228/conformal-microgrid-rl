"""
Sim-to-Real 評估指標計算模組

提供兩個主要功能：
1. SOC軌跡的Sim-to-Real擬合度評估
2. 電流控制的平滑性與安全性評估
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from scipy import stats
from scipy.spatial.distance import euclidean
try:
    from dtaidistance import dtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    print("Warning: dtaidistance not installed. DTW metrics will be unavailable. Install with: pip install dtaidistance")


def compute_soc_sim2real_fitness(
    sim_soc_trajectories: Union[List[List[float]], np.ndarray],
    real_soc_data: Union[pd.DataFrame, np.ndarray, str],
    sim_time_step: float = 1.0,
    real_timestamp_col: str = 'timestamp',
    real_soc_col: str = 'soc_percent',
    alignment_method: str = 'interpolate',
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    計算SOC軌跡的Sim-to-Real擬合度
    
    Parameters:
    -----------
    sim_soc_trajectories : List[List[float]] or np.ndarray
        仿真SOC軌跡，每個元素是一個episode的SOC序列
        Shape: (n_episodes, episode_length) 或 (episode_length,)
    real_soc_data : pd.DataFrame, np.ndarray, or str (path to CSV)
        實際SOC數據
        - DataFrame: 必須包含timestamp和soc_percent列
        - ndarray: Shape (n_samples, 2)，第一列為時間，第二列為SOC
        - str: CSV文件路徑
    sim_time_step : float
        仿真時間步長（小時），預設1.0
    real_timestamp_col : str
        實際數據的時間戳列名
    real_soc_col : str
        實際數據的SOC列名
    alignment_method : str
        時間對齊方法：'interpolate' (線性插值) 或 'resample' (重採樣)
    metrics : List[str], optional
        要計算的指標列表，預設為全部
        可選：'mse', 'mae', 'rmse', 'r2', 'correlation', 'dtw', 'max_error'
    
    Returns:
    --------
    Dict[str, float]
        包含各種擬合度指標的字典
    """
    # 預設指標列表
    if metrics is None:
        metrics = ['mse', 'mae', 'rmse', 'r2', 'correlation', 'max_error']
        if HAS_DTW:
            metrics.append('dtw')
    
    # 處理輸入數據
    sim_soc = np.asarray(sim_soc_trajectories)
    if sim_soc.ndim == 1:
        sim_soc = sim_soc.reshape(1, -1)
    elif sim_soc.ndim > 2:
        raise ValueError(f"sim_soc_trajectories must be 1D or 2D, got {sim_soc.ndim}D")
    
    # 處理實際數據
    if isinstance(real_soc_data, str):
        real_df = pd.read_csv(real_soc_data)
    elif isinstance(real_soc_data, pd.DataFrame):
        real_df = real_soc_data.copy()
    elif isinstance(real_soc_data, np.ndarray):
        real_df = pd.DataFrame({
            real_timestamp_col: real_soc_data[:, 0],
            real_soc_col: real_soc_data[:, 1]
        })
    else:
        raise TypeError(f"real_soc_data must be DataFrame, ndarray, or str, got {type(real_soc_data)}")
    
    # 確保時間戳是數值型或可轉換為datetime
    if real_df[real_timestamp_col].dtype == 'object':
        try:
            real_df[real_timestamp_col] = pd.to_datetime(real_df[real_timestamp_col])
        except:
            real_df[real_timestamp_col] = pd.to_numeric(real_df[real_timestamp_col], errors='coerce')
    
    # 轉換為數值時間（從開始的相對時間，單位：小時）
    if pd.api.types.is_datetime64_any_dtype(real_df[real_timestamp_col]):
        real_df['time_hours'] = (real_df[real_timestamp_col] - real_df[real_timestamp_col].min()).dt.total_seconds() / 3600.0
    else:
        real_df['time_hours'] = (real_df[real_timestamp_col] - real_df[real_timestamp_col].min()) / 3600.0
    
    # 提取實際SOC
    real_soc = real_df[real_soc_col].values
    real_time = real_df['time_hours'].values
    
    # 對齊時間序列
    results = {}
    
    # 對每個仿真episode計算指標
    episode_metrics = []
    for i, sim_soc_ep in enumerate(sim_soc):
        # 生成仿真時間
        sim_time = np.arange(len(sim_soc_ep)) * sim_time_step
        
        # 時間對齊
        if alignment_method == 'interpolate':
            # 線性插值：將實際數據插值到仿真時間點
            aligned_real = np.interp(sim_time, real_time, real_soc)
            aligned_sim = sim_soc_ep
        elif alignment_method == 'resample':
            # 重採樣：將仿真數據重採樣到實際時間點
            aligned_sim = np.interp(real_time, sim_time, sim_soc_ep)
            aligned_real = real_soc
        else:
            raise ValueError(f"Unknown alignment_method: {alignment_method}")
        
        # 計算指標
        ep_metrics = {}
        
        if 'mse' in metrics:
            ep_metrics['mse'] = np.mean((aligned_sim - aligned_real) ** 2)
        
        if 'mae' in metrics:
            ep_metrics['mae'] = np.mean(np.abs(aligned_sim - aligned_real))
        
        if 'rmse' in metrics:
            ep_metrics['rmse'] = np.sqrt(np.mean((aligned_sim - aligned_real) ** 2))
        
        if 'r2' in metrics:
            ss_res = np.sum((aligned_real - aligned_sim) ** 2)
            ss_tot = np.sum((aligned_real - np.mean(aligned_real)) ** 2)
            if ss_tot > 1e-10:
                ep_metrics['r2'] = 1 - (ss_res / ss_tot)
            else:
                ep_metrics['r2'] = 0.0
        
        if 'correlation' in metrics:
            if len(aligned_sim) > 1 and np.std(aligned_sim) > 1e-10 and np.std(aligned_real) > 1e-10:
                corr, _ = stats.pearsonr(aligned_sim, aligned_real)
                ep_metrics['correlation'] = corr
            else:
                ep_metrics['correlation'] = 0.0
        
        if 'max_error' in metrics:
            ep_metrics['max_error'] = np.max(np.abs(aligned_sim - aligned_real))
        
        if 'dtw' in metrics and HAS_DTW:
            # Dynamic Time Warping 距離
            try:
                dtw_distance = dtw.distance(aligned_sim, aligned_real)
                # 正規化到 [0, 1] 區間（使用最大可能距離）
                max_possible_distance = np.sqrt(len(aligned_sim)) * (np.max(aligned_real) - np.min(aligned_real))
                if max_possible_distance > 1e-10:
                    ep_metrics['dtw'] = dtw_distance / max_possible_distance
                else:
                    ep_metrics['dtw'] = 0.0
            except:
                ep_metrics['dtw'] = np.nan
        
        episode_metrics.append(ep_metrics)
    
    # 聚合所有episodes的指標
    for metric_name in metrics:
        if metric_name == 'dtw' and not HAS_DTW:
            continue
        
        values = [ep_metrics.get(metric_name, np.nan) for ep_metrics in episode_metrics]
        values = [v for v in values if not np.isnan(v)]
        
        if len(values) > 0:
            results[f'{metric_name}_mean'] = np.mean(values)
            results[f'{metric_name}_std'] = np.std(values)
            results[f'{metric_name}_min'] = np.min(values)
            results[f'{metric_name}_max'] = np.max(values)
        else:
            results[f'{metric_name}_mean'] = np.nan
            results[f'{metric_name}_std'] = np.nan
            results[f'{metric_name}_min'] = np.nan
            results[f'{metric_name}_max'] = np.nan
    
    # 添加總體統計
    results['n_episodes'] = len(sim_soc)
    results['alignment_method'] = alignment_method
    
    return results


def compute_current_smoothness_safety(
    current_data: Union[pd.DataFrame, np.ndarray, str],
    current_col: str = 'current_a',
    timestamp_col: str = 'timestamp',
    time_step: Optional[float] = None,
    max_current: Optional[float] = None,
    safety_threshold_ratio: Optional[float] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    計算電流控制的平滑性與安全性指標
    
    Parameters:
    -----------
    current_data : pd.DataFrame, np.ndarray, or str (path to CSV)
        電流數據
        - DataFrame: 必須包含current_col列
        - ndarray: 1D數組或2D數組（第一列為時間，第二列為電流）
        - str: CSV文件路徑
    current_col : str
        電流列名（DataFrame時使用）
    timestamp_col : str
        時間戳列名（DataFrame時使用）
    time_step : float, optional
        時間步長（小時），如果不提供則從時間戳計算
    max_current : float, optional
        最大允許電流（A），用於安全性檢查
    safety_threshold_ratio : float, optional
        安全閾值（相對max_current的比例），預設0.9
    metrics : List[str], optional
        要計算的指標列表，預設為全部
        可選：'smoothness', 'variance', 'max_change_rate', 'std', 
              'overcurrent_events', 'safety_ratio', 'jerk', 'rms'
    
    Returns:
    --------
    Dict[str, float]
        包含各種平滑性和安全性指標的字典
    """
    # 預設指標列表
    if metrics is None:
        metrics = ['smoothness', 'variance', 'std', 'max_change_rate', 'rms', 'jerk']
        if max_current is not None:
            metrics.extend(['overcurrent_events', 'safety_ratio'])
    
    # 處理輸入數據
    if isinstance(current_data, str):
        df = pd.read_csv(current_data)
    elif isinstance(current_data, pd.DataFrame):
        df = current_data.copy()
    elif isinstance(current_data, np.ndarray):
        if current_data.ndim == 1:
            current_array = current_data
            time_array = None
        elif current_data.ndim == 2 and current_data.shape[1] >= 2:
            time_array = current_data[:, 0]
            current_array = current_data[:, 1]
        else:
            raise ValueError(f"current_data ndarray must be 1D or 2D with shape (n, 2), got {current_data.shape}")
        
        df = pd.DataFrame({current_col: current_array})
        if time_array is not None:
            df[timestamp_col] = time_array
    else:
        raise TypeError(f"current_data must be DataFrame, ndarray, or str, got {type(current_data)}")
    
    # 提取電流數據
    current = df[current_col].values
    current = current[~np.isnan(current)]  # 移除NaN
    
    if len(current) < 2:
        raise ValueError(f"Need at least 2 data points, got {len(current)}")
    
    results = {}
    
    # 計算時間步長
    if time_step is None:
        if timestamp_col in df.columns:
            timestamps = df[timestamp_col].values
            if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                time_diffs = pd.Series(timestamps).diff().dt.total_seconds() / 3600.0
            else:
                time_diffs = pd.Series(timestamps).diff() / 3600.0
            time_diffs = time_diffs.dropna().values
            if len(time_diffs) > 0:
                time_step = np.median(time_diffs)
            else:
                time_step = 1.0  # 預設1小時
        else:
            time_step = 1.0  # 預設1小時
    
    # 平滑性指標
    if 'variance' in metrics:
        results['variance'] = np.var(current)
    
    if 'std' in metrics:
        results['std'] = np.std(current)
    
    if 'rms' in metrics:
        results['rms'] = np.sqrt(np.mean(current ** 2))
    
    # 變化率（一階導數）
    if 'max_change_rate' in metrics or 'smoothness' in metrics or 'jerk' in metrics:
        current_diff = np.diff(current)
        change_rate = current_diff / time_step  # A/h
    
    if 'max_change_rate' in metrics:
        results['max_change_rate'] = np.max(np.abs(change_rate))
        results['mean_change_rate'] = np.mean(np.abs(change_rate))
        results['std_change_rate'] = np.std(change_rate)
    
    # 平滑性指標（基於變化率的倒數）
    if 'smoothness' in metrics:
        # 平滑性 = 1 / (1 + 變化率標準差)
        # 值越大越平滑，範圍 [0, 1]
        change_rate_std = np.std(change_rate) if len(change_rate) > 0 else 0.0
        if change_rate_std > 0:
            results['smoothness'] = 1.0 / (1.0 + change_rate_std)
        else:
            results['smoothness'] = 1.0
    
    # Jerk（三階導數，衡量加速度變化率）
    if 'jerk' in metrics:
        if len(current) >= 3:
            current_diff2 = np.diff(current_diff)
            jerk = current_diff2 / (time_step ** 2)  # A/h²
            results['jerk_mean'] = np.mean(np.abs(jerk))
            results['jerk_max'] = np.max(np.abs(jerk))
            results['jerk_std'] = np.std(jerk)
        else:
            results['jerk_mean'] = 0.0
            results['jerk_max'] = 0.0
            results['jerk_std'] = 0.0
    
    # 安全性指標
    if max_current is not None:
        safety_threshold_ratio_val = safety_threshold_ratio if safety_threshold_ratio is not None else 0.9
        safety_threshold_val = safety_threshold_ratio_val * max_current
        
        if 'overcurrent_events' in metrics:
            # 過流事件數
            overcurrent_mask = np.abs(current) > max_current
            results['overcurrent_events'] = np.sum(overcurrent_mask)
            results['overcurrent_ratio'] = np.sum(overcurrent_mask) / len(current)
        
        if 'safety_ratio' in metrics:
            # 安全操作比例（在安全閾值內的時間比例）
            safe_mask = np.abs(current) <= safety_threshold_val
            results['safety_ratio'] = np.sum(safe_mask) / len(current)
            results['safety_threshold_used'] = safety_threshold_val
        
        if 'max_current_usage' in metrics:
            # 最大電流使用率
            results['max_current_usage'] = np.max(np.abs(current)) / max_current
            results['mean_current_usage'] = np.mean(np.abs(current)) / max_current
    
    # 統計信息
    results['n_samples'] = len(current)
    results['time_step_hours'] = time_step
    results['current_mean'] = np.mean(current)
    results['current_min'] = np.min(current)
    results['current_max'] = np.max(current)
    results['current_range'] = np.max(current) - np.min(current)
    
    return results


def load_sim_results(sim_results_path: str) -> Dict:
    """
    載入仿真結果JSON文件
    
    Parameters:
    -----------
    sim_results_path : str
        仿真結果JSON文件路徑
    
    Returns:
    --------
    Dict
        包含episode_soc_trajectories等的字典
    """
    with open(sim_results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def compute_all_metrics(
    sim_results_path: str,
    real_data_path: str,
    sim_time_step: float = 1.0,
    max_current: Optional[float] = None,
    output_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    計算所有Sim-to-Real指標
    
    Parameters:
    -----------
    sim_results_path : str
        仿真結果JSON文件路徑
    real_data_path : str
        實際數據CSV文件路徑
    sim_time_step : float
        仿真時間步長（小時）
    max_current : float, optional
        最大允許電流（A）
    output_path : str, optional
        輸出JSON文件路徑
    
    Returns:
    --------
    Dict[str, Dict[str, float]]
        包含所有指標的字典
    """
    # 載入仿真結果
    sim_results = load_sim_results(sim_results_path)
    
    # 計算SOC擬合度
    soc_trajectories = sim_results.get('episode_soc_trajectories', [])
    if len(soc_trajectories) == 0:
        raise ValueError("No episode_soc_trajectories found in simulation results")
    
    soc_metrics = compute_soc_sim2real_fitness(
        soc_trajectories,
        real_data_path,
        sim_time_step=sim_time_step
    )
    
    # 計算電流平滑性和安全性
    current_metrics = compute_current_smoothness_safety(
        real_data_path,
        max_current=max_current
    )
    
    # 組合結果
    all_metrics = {
        'soc_sim2real_fitness': soc_metrics,
        'current_smoothness_safety': current_metrics
    }
    
    # 保存結果
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {output_path}")
    
    return all_metrics


if __name__ == '__main__':
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute Sim-to-Real metrics')
    parser.add_argument('--sim', type=str, required=True, help='Simulation results JSON path')
    parser.add_argument('--real', type=str, required=True, help='Real data CSV path')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--time-step', type=float, default=1.0, help='Simulation time step (hours)')
    parser.add_argument('--max-current', type=float, default=None, help='Maximum allowed current (A)')
    
    args = parser.parse_args()
    
    metrics = compute_all_metrics(
        args.sim,
        args.real,
        sim_time_step=args.time_step,
        max_current=args.max_current,
        output_path=args.output
    )
    
    print("\n=== SOC Sim-to-Real Fitness Metrics ===")
    for key, value in metrics['soc_sim2real_fitness'].items():
        print(f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}")
    
    print("\n=== Current Smoothness & Safety Metrics ===")
    for key, value in metrics['current_smoothness_safety'].items():
        print(f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}")
