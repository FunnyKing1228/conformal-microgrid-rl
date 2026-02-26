import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'safe_fl_microgrid'))

import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sac_agent import SACAgent
from microgrid_env import create_microgrid_env, MicrogridEnvironment
from experiment_manager import ExperimentManager, create_experiment_from_config
from compute_resources import collect_compute_resources, format_compute_resources
import time
from typing import List, Dict, Any
import argparse
from safety_net import project as safety_project, update_conformal_residual, set_conformal_params, clear_residual_buffer, get_residual_count


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_agent(config: Dict[str, Any], state_dim: int, action_dim: int, device: str) -> SACAgent:
    """Create SAC agent from configuration"""
    sac_config = config['sac']
    training_cfg = config.get('training', {})
    variant = training_cfg.get('variant', 'sac')
    evidential_enabled = variant == 'sac_sn_evi'
    lambda_evi = float(training_cfg.get('lambda_evi', 1e-3))
    beta_risk = float(training_cfg.get('beta_risk', 0.5))
    target_entropy = sac_config.get('target_entropy', -1.0)
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr_actor=sac_config['actor_lr'],
        lr_critic=sac_config['critic_lr'],
        gamma=sac_config['gamma'],
        tau=sac_config['tau'],
        alpha=sac_config['alpha'],
        target_entropy=target_entropy,
        hidden_dim=sac_config['hidden_dim'],
        buffer_size=sac_config['buffer_size'],
        batch_size=sac_config['batch_size'],
        evidential_enabled=evidential_enabled,
        lambda_evi=lambda_evi,
        beta_risk=beta_risk
    )
    
    return agent


def create_environment(config: Dict[str, Any]) -> MicrogridEnvironment:
    """Create microgrid environment from configuration"""
    env_config = config['env']
    safetynet_cfg = config.get('safetynet', {})
    stress_cfg = config.get('stress', {})
    
    # 環境端不啟用 ramp（避免與 SafetyNet 重複），hard_guard 由 env 配置決定
    # 準備 stress 參數（對應 MicrogridEnvironment __init__ 的 stress_*）
    stress_kwargs = {}
    if stress_cfg:
        stress_kwargs = {
            'stress_enable': bool(stress_cfg.get('enable', False)),
            'stress_efficiency_noise_std': float(stress_cfg.get('efficiency_noise_std', 0.0)),
            'stress_dt_jitter_std': float(stress_cfg.get('dt_jitter_std', 0.0)),
            'stress_action_lag_alpha': float(stress_cfg.get('action_lag_alpha', 0.0)),
            'stress_soc_obs_delay': int(stress_cfg.get('soc_obs_delay', 0)),
            'stress_soc_obs_noise_std': float(stress_cfg.get('soc_obs_noise_std', 0.0)),
            'stress_bounds_drift_std': float(stress_cfg.get('bounds_drift_std', 0.0)),
            'stress_external_pmax_shrink_prob': float(stress_cfg.get('external_pmax_shrink_prob', 0.0)),
            'stress_external_pmax_shrink_factor': float(stress_cfg.get('external_pmax_shrink_factor', 1.0)),
            'stress_power_loss_ratio': float(stress_cfg.get('power_loss_ratio', 0.0)),
        }

    env = create_microgrid_env(
        microgrid_id=env_config.get('microgrid_id', 0),
        episode_length=env_config['episode_length'],
        battery_capacity_kwh=env_config['battery_capacity_kwh'],
        battery_power_kw=env_config['battery_power_kw'],
        use_real_data=env_config['use_real_data'],
        time_step=env_config.get('time_step', 1.0),
        ramp_limit_kw=env_config.get('ramp_limit_kw', None),
        hard_guard=env_config.get('hard_guard', False),
        dataset_csv_path=env_config.get('dataset_csv_path', None),
        dataset_pv_join_wind=env_config.get('dataset_pv_join_wind', False),
        train_window_hours=env_config.get('train_window_hours', None),
        dataset_pv_column=env_config.get('dataset_pv_column', None),
        dataset_load_kw=env_config.get('dataset_load_kw', None),
        dataset_power_scale=env_config.get('dataset_power_scale', 1.0),
        dataset_time_column=env_config.get('dataset_time_column', None),
        synthetic_hourly_hold=env_config.get('synthetic_hourly_hold', False),
        synthetic_pv_peak_kw=env_config.get('synthetic_pv_peak_kw', 20.0),
        synthetic_pv_start_hour=env_config.get('synthetic_pv_start_hour', 6),
        synthetic_pv_end_hour=env_config.get('synthetic_pv_end_hour', 18),
        synthetic_load_base_kw=env_config.get('synthetic_load_base_kw', 10.0),
        synthetic_load_amp_kw=env_config.get('synthetic_load_amp_kw', 5.0),
        synthetic_price_base=env_config.get('synthetic_price_base', 0.12),
        synthetic_price_peak=env_config.get('synthetic_price_peak', 0.20),
        synthetic_price_peak_start=env_config.get('synthetic_price_peak_start', 8),
        synthetic_price_peak_end=env_config.get('synthetic_price_peak_end', 18),
        allow_grid_trading=env_config.get('allow_grid_trading', True),  # 預設允許電網交易
        **stress_kwargs
    )
    # 將 SafetyNet 的 ramp 參數掛在 env 上，供訓練與評估共用
    try:
        env.safetynet_ramp_kw = config.get('safetynet', {}).get('ramp_limit_kw', None)
    except Exception:
        env.safetynet_ramp_kw = None
    # 將可配置的懲罰注入環境，避免 reward_scaling 稀釋後過小
    reward_cfg = config.get('reward', {})
    try:
        env.realized_violation_penalty = float(reward_cfg.get('realized_violation_penalty', 20.0))
    except Exception:
        env.realized_violation_penalty = 20.0
    
    return env


def train_sac_with_microgrid(
    env: MicrogridEnvironment,
    agent: SACAgent,
    config: Dict[str, Any],
    exp_manager: ExperimentManager
) -> Dict[str, List[float]]:
    """
    Train SAC agent using microgrid environment
    
    Args:
        env: Microgrid environment
        agent: SAC agent
        config: Training configuration
        exp_manager: Experiment manager for organizing outputs
    
    Returns:
        Dictionary containing training metrics
    """
    
    training_config = config['training']
    logging_config = config['logging']
    
    total_episodes = training_config['total_episodes']
    max_steps = training_config['max_steps']
    update_every = config['sac']['update_every']
    eval_every = training_config['eval_every']
    save_every = training_config['save_every']
    eval_episodes = training_config['eval_episodes']
    log_interval = logging_config['log_interval']
    warmup_steps = config['sac']['warmup_steps']
    # 讀取 variant 開關（預設 sac 作為 Baseline）
    variant = config.get('training', {}).get('variant', 'sac')
    use_safetynet = variant in ('sac_sn', 'sac_sn_evi')

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_soc_violations = []
    episode_action_violations = []
    episode_actions = []
    actions_raw_series = []
    actions_safe_series = []
    episode_soc_trajectories = []
    episode_revenues = []
    episode_costs = []
    # 新增KPI：每集嘗試違規/投影介入/實際違規的計數（皆為每步布林計數的總和）
    episode_attempted_violations = []
    episode_safety_projected = []
    episode_realized_violations = []
    
    # Evaluation metrics
    eval_rewards = []
    eval_soc_violations = []
    eval_revenues = []
    eval_costs = []
    
    best_eval_reward = float('-inf')
    # Adaptive conformal/penalty parameters
    conformal_cfg = config.get('conformal', {})
    current_conformal_window = int(conformal_cfg.get('window', 2880))
    current_conformal_delta = float(conformal_cfg.get('delta', 0.1))
    projected_penalty_mult = 1.0
    
    print(f"Starting SAC training with Microgrid Environment for {total_episodes} episodes...")
    print(f"Device: {agent.device}")
    print(f"Update every: {update_every} steps")
    print(f"Evaluation every: {eval_every} episodes")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Experiment directory: {exp_manager.experiment_dir}")
    print(f"Environment: {env.__class__.__name__}")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    for episode in range(total_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        soc_violations = 0
        action_violations = 0
        actions = []  # legacy avg (will use normalized)
        actions_raw = []       # normalized [-1,1]
        actions_safe = []      # normalized [-1,1]
        actions_safe_kw = []   # kW，用於 prev_action 與分析
        soc_trajectory = [state[0]]  # First element is SoC
        episode_revenue = 0.0
        episode_cost = 0.0
        # 每集KPI累積器
        attempted_count = 0
        projected_count = 0
        realized_count = 0
        # 追蹤上一個步驟的環境累積違規次數（用以計算本步是否實際違規）
        prev_soc_violations_cum = 0
        prev_action_violations_cum = 0

        for step in range(max_steps):
            # Select action
            action_norm = agent.select_action(state, evaluate=False)  # [-1, 1]
            action_norm_val = float(action_norm[0])
            actions.append(action_norm_val)
            actions_raw.append(action_norm_val)
            power_scale = float(getattr(env, 'battery_power_kw', 1.0))
            a_raw_kw = action_norm_val * power_scale
            
            # 影子模式：以Raw動作估計下一步SoC，判斷是否「嘗試違規」（不影響控制）
            try:
                current_soc = float(state[0])
                soc_next_raw = float(env.predict_soc_raw(current_soc, a_raw_kw)) if hasattr(env, 'predict_soc_raw') else None
                soc_min = float(getattr(env, 'soc_min_eff', getattr(env, 'soc_min', 0.0)))
                soc_max = float(getattr(env, 'soc_max_eff', getattr(env, 'soc_max', 1.0)))
                attempted = int(soc_next_raw is not None and (soc_next_raw < soc_min or soc_next_raw > soc_max))
            except Exception:
                attempted = 0
            attempted_count += attempted

            # SafetyNet 投影：僅在 use_safetynet=True 時啟用；純 SAC 直接用 raw 動作
            pmax = power_scale
            if use_safetynet:
                ramp_kw = getattr(env, 'safetynet_ramp_kw', None)
                soc_bounds = (
                    float(getattr(env, 'soc_min_eff', getattr(env, 'soc_min', 0.0))),
                    float(getattr(env, 'soc_max_eff', getattr(env, 'soc_max', 1.0)))
                )
                prev_a_kw = float(actions_safe_kw[-1]) if actions_safe_kw else 0.0
                a_safe_kw, did_project, delta_kw = safety_project(
                    state=state,
                    action=np.array([a_raw_kw], dtype=np.float32),
                    prev_action=prev_a_kw,
                    pmax=pmax,
                    ramp_kw=ramp_kw,
                    soc_bounds=soc_bounds,
                    env=env,
                )
                safety_projected = int(did_project)
                projected_count += safety_projected
                a_safe_norm = max(-1.0, min(1.0, a_safe_kw / power_scale if power_scale > 0 else 0.0))
                actions_safe.append(a_safe_norm)
                actions_safe_kw.append(a_safe_kw)
            else:
                # Baseline SAC：不使用 SafetyNet，動作不投影
                delta_kw = 0.0
                safety_projected = 0
                a_safe_kw = a_raw_kw
                a_safe_norm = action_norm_val
                actions_safe.append(a_safe_norm)
                actions_safe_kw.append(a_safe_kw)

            # 預測下一步 SoC 用於 conformal 殘差
            soc_pred_next = None
            try:
                if hasattr(env, 'predict_soc_raw'):
                    soc_pred_next = float(env.predict_soc_raw(float(state[0]), float(a_safe_kw)))
            except Exception:
                soc_pred_next = None

            # Take action
            next_state, reward, terminated, truncated, step_info = env.step([a_safe_kw])
            # Apply safety shaping only for SafetyNet variants, and scale consistently with env reward
            # Note: env reward already multiplied by reward_scaling; we scale penalties by the same factor
            scale_guard = max(float(getattr(env, 'reward_scaling', 1.0)), 1e-9)
            if use_safetynet:
                attempted_penalty_val = float(config.get('reward', {}).get('attempted_violation_penalty', 0.1)) if attempted else 0.0
                # 投影懲罰按幅度歸一化到 Pmax
                pmax_guard = max(pmax, 1e-9)
                proj_unit = float(config.get('reward', {}).get('safety_projection_penalty', 0.001))
                projected_penalty_val = projected_penalty_mult * proj_unit * (float(delta_kw) / pmax_guard) if safety_projected else 0.0
                reward -= scale_guard * (attempted_penalty_val + projected_penalty_val)
            done = terminated or truncated
            
            # Store transition（帶入 OCC 代理成本 Δa/Pmax）
            occ_proxy = float(delta_kw) / max(pmax, 1e-9)
            agent.store_transition(state, np.array([a_safe_norm], dtype=np.float32), reward, next_state, done, occ_proxy=occ_proxy)
            
            # Update networks (only after warmup)
            if (step % update_every == 0 and 
                len(agent.replay_buffer) >= warmup_steps and
                len(agent.replay_buffer) >= agent.batch_size):
                update_info = agent.update()
                if episode % log_interval == 0 and step == 0:
                    print(f"Episode {episode}, Step {step}: {update_info}")
            
            # Update episode metrics
            episode_reward += reward
            episode_length += 1
            soc_trajectory.append(next_state[0])
            # 更新 conformal 殘差
            try:
                if soc_pred_next is not None:
                    update_conformal_residual(float(next_state[0]) - float(soc_pred_next))
            except Exception:
                pass
            
            # Update violation counts（以環境回傳的累積值做差分）
            current_soc_violations_cum = int(step_info.get('soc_violations', 0))
            realized = max(0, current_soc_violations_cum - prev_soc_violations_cum)
            prev_soc_violations_cum = current_soc_violations_cum
            realized_count += realized
            soc_violations += realized

            current_action_violations_cum = int(step_info.get('action_violations', 0))
            action_violation_step = max(0, current_action_violations_cum - prev_action_violations_cum)
            prev_action_violations_cum = current_action_violations_cum
            action_violations += action_violation_step
 
            # Remove duplicate per-episode penalty application; shaping already applied above when enabled

            # Update financial metrics
            episode_revenue += step_info.get('total_revenue', 0)
            episode_cost += step_info.get('total_cost', 0)
            
            if done:
                break
                
            state = next_state
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_soc_violations.append(soc_violations)
        episode_action_violations.append(action_violations)
        episode_actions.append(np.mean(np.abs(actions)))
        actions_raw_series.append(np.mean(np.abs(actions_raw)) if actions_raw else 0.0)
        actions_safe_series.append(np.mean(np.abs(actions_safe)) if actions_safe else 0.0)
        episode_soc_trajectories.append(soc_trajectory)
        episode_revenues.append(episode_revenue)
        episode_costs.append(episode_cost)
        # 新增KPI彙總
        episode_attempted_violations.append(attempted_count)
        episode_safety_projected.append(projected_count)
        episode_realized_violations.append(realized_count)
        
        # 自適應調整：根據本集 realized 調整 conformal delta 與投影懲罰倍率
        try:
            target_low, target_high = 5, 10
            if realized_count > target_high:
                # Tighten tube,增加投影懲罰影響
                current_conformal_delta = max(0.01, current_conformal_delta - 0.005)
                projected_penalty_mult = min(2.0, projected_penalty_mult * 1.1)
                set_conformal_params(window=current_conformal_window, delta=current_conformal_delta)
            elif realized_count <= target_low:
                # 放鬆 tube、減少投影懲罰
                current_conformal_delta = min(0.15, current_conformal_delta + 0.005)
                projected_penalty_mult = max(0.5, projected_penalty_mult * 0.95)
                set_conformal_params(window=current_conformal_window, delta=current_conformal_delta)
        except Exception:
            pass

        # Evaluation
        if episode % eval_every == 0:
            eval_reward, eval_violations, eval_revenue, eval_cost = evaluate_microgrid_agent(
                agent, env, n_episodes=eval_episodes, use_safetynet=use_safetynet,
                stress_eval_seed=training_config.get('stress_eval_seed', None)
            )
            eval_rewards.append(eval_reward)
            eval_soc_violations.append(eval_violations)
            eval_revenues.append(eval_revenue)
            eval_costs.append(eval_cost)
            
            print(f"Episode {episode:4d} | "
                  f"Train Reward: {episode_reward:6.2f} | "
                  f"Eval Reward: {eval_reward:6.2f} | "
                  f"Eval Violations: {eval_violations:2d} | "
                  f"Eval Revenue: ${eval_revenue:5.2f} | "
                  f"Eval Cost: ${eval_cost:5.2f} | "
                  f"Buffer Size: {len(agent.replay_buffer)}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                if logging_config['save_models']:
                    # Save to experiment directory
                    best_model_path = os.path.join(exp_manager.models_dir, "best_sac_model.pth")
                    agent.save(best_model_path)
                    # Suppress verbose: best model silently saved
        
        # Save checkpoint
        if episode % save_every == 0 and logging_config['save_models']:
            checkpoint_path = os.path.join(exp_manager.models_dir, f"sac_checkpoint_ep{episode}.pth")
            agent.save(checkpoint_path)
        
        # Progress logging
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_violations = np.mean(episode_soc_violations[-log_interval:])
            avg_revenue = np.mean(episode_revenues[-log_interval:])
            avg_cost = np.mean(episode_costs[-log_interval:])
            print(f"Episode {episode:4d} | "
                  f"Avg Reward (last {log_interval}): {avg_reward:6.2f} | "
                  f"Avg Violations: {avg_violations:4.2f} | "
                  f"Avg Revenue: ${avg_revenue:5.2f} | "
                  f"Avg Cost: ${avg_cost:5.2f}")
    
    # Save final model
    if logging_config['save_models']:
        final_model_path = os.path.join(exp_manager.models_dir, "final_sac_model.pth")
        agent.save(final_model_path)
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_soc_violations': episode_soc_violations,
        'episode_action_violations': episode_action_violations,
        'episode_actions': episode_actions,
        'episode_actions_raw': actions_raw_series,
        'episode_actions_safe': actions_safe_series,
        'episode_soc_trajectories': episode_soc_trajectories,
        'episode_revenues': episode_revenues,
        'episode_costs': episode_costs,
        # 新增KPI輸出
        'episode_attempted_violations': episode_attempted_violations,
        'episode_safety_projected': episode_safety_projected,
        'episode_realized_violations': episode_realized_violations,
        'eval_rewards': eval_rewards,
        'eval_soc_violations': eval_soc_violations,
        'eval_revenues': eval_revenues,
        'eval_costs': eval_costs
    }


def evaluate_microgrid_agent(agent: SACAgent, env: MicrogridEnvironment, n_episodes: int = 5, n_steps: int | None = None, use_safetynet: bool = False, stress_eval_seed: int | None = None) -> tuple:
    """Evaluate agent over multiple episodes"""
    total_reward = 0
    total_violations = 0
    total_revenue = 0
    total_cost = 0
    if n_steps is None:
        n_steps = getattr(env, 'episode_length', 24)
    # 評估冷啟動：重設 conformal 視窗並清空殘差
    try:
        conformal_window = int(getattr(env, 'episode_length', n_steps)) * 2 if n_steps is not None else 2880
        # 保持當前 delta；僅確保窗口合理，並清空緩衝
        set_conformal_params(window=conformal_window, delta=float(0.1))
        clear_residual_buffer()
        print(f"[EVAL] conformal residuals cleared; count={get_residual_count()}")
    except Exception:
        pass
    # 評估擾動 seed（若提供）
    if stress_eval_seed is not None:
        try:
            np.random.seed(int(stress_eval_seed))
        except Exception:
            pass
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        # 若環境支援固定驗證窗，鎖定起點為末 720h
        try:
            if hasattr(env, 'episode_data') and hasattr(env, 'load_data'):
                # 固定驗證窗：最後 n_steps 視窗
                total_len = int(min(len(env.load_data), len(env.pv_data), len(env.price_data)))
                if n_steps is not None and total_len >= int(n_steps):
                    start_idx_eval = int(total_len - int(n_steps))
                    setattr(env, 'fixed_start_idx', start_idx_eval)
                    state, _ = env.reset()
        except Exception:
            pass
        episode_reward = 0
        episode_violations = 0
        episode_revenue = 0
        episode_cost = 0
        prev_soc_violations_cum = 0
        prev_a_eval = 0.0
        
        for step in range(int(n_steps)):
            action = agent.select_action(state, evaluate=True)  # [-1, 1]
            pmax = float(getattr(env, 'battery_power_kw', 0.0))
            a_raw_kw = float(action[0]) * pmax
            if use_safetynet:
                ramp_kw = getattr(env, 'safetynet_ramp_kw', None)
                soc_bounds = (float(getattr(env, 'soc_min', 0.0)), float(getattr(env, 'soc_max', 1.0)))
                a_safe_kw, _, _ = safety_project(
                    state=state,
                    action=np.array([a_raw_kw], dtype=np.float32),
                    prev_action=prev_a_eval,  # 評估也維持 ramp 連續性
                    pmax=pmax,
                    ramp_kw=ramp_kw,
                    soc_bounds=soc_bounds,
                    env=env,
                )
                # 評估期開啟硬護欄
                need_reset = False
                if hasattr(env, 'hard_guard') and not bool(getattr(env, 'hard_guard')):
                    setattr(env, 'hard_guard', True)
                    need_reset = True
                next_state, reward, terminated, truncated, info = env.step([a_safe_kw])
                if need_reset:
                    setattr(env, 'hard_guard', False)
                prev_a_eval = a_safe_kw
            else:
                next_state, reward, terminated, truncated, info = env.step([a_raw_kw])
            done = terminated or truncated
            
            episode_reward += reward
            # 以累積違規的差分計算當步是否違規
            current_soc_violations_cum = int(info.get('soc_violations', 0))
            episode_violations += max(0, current_soc_violations_cum - prev_soc_violations_cum)
            prev_soc_violations_cum = current_soc_violations_cum
            episode_revenue += info.get('total_revenue', 0)
            episode_cost += info.get('total_cost', 0)
            
            if done:
                break
                
            state = next_state
 
        # 釋放固定起點，避免影響下一回合訓練
        try:
            if hasattr(env, 'fixed_start_idx'):
                setattr(env, 'fixed_start_idx', None)
        except Exception:
            pass
        total_reward += episode_reward
        total_violations += episode_violations
        total_revenue += episode_revenue
        total_cost += episode_cost
 
    return (total_reward / n_episodes, total_violations, 
            total_revenue / n_episodes, total_cost / n_episodes)


def plot_microgrid_training_results(metrics: Dict[str, List[float]], config: Dict[str, Any], 
                                   exp_manager: ExperimentManager, save_path: str = "training_results.png"):
    """Plot training results for microgrid environment"""
    # 空資料保護：若沒有任何 episode，產生診斷圖而非空白圖
    try:
        num_eps_guard = int(len(metrics.get('episode_rewards', [])))
    except Exception:
        num_eps_guard = 0
    if num_eps_guard == 0:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.axis('off')
        present_keys = list(metrics.keys()) if isinstance(metrics, dict) else []
        msg_lines = [
            'No training data available to plot.',
            'This diagnostic page is generated to avoid a blank image.',
            f"Available keys: {present_keys}",
            'Expected keys include: episode_rewards, episode_lengths, episode_realized_violations,',
            'episode_attempted_violations, episode_safety_projected, episode_revenues, episode_costs.'
        ]
        ax.text(0.02, 0.98, "\n".join(msg_lines), va='top', ha='left', fontsize=12)
        plt.tight_layout()
        exp_plot_path = os.path.join(exp_manager.results_dir, "training_results.png")
        plt.savefig(exp_plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Episode rewards
    axes[0, 0].plot(metrics['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(metrics['episode_lengths'])
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # SoC violations
    # 使用已校正的每集實際違規次數（若不存在則退回原欄位）
    soc_series = metrics.get('episode_realized_violations') or metrics.get('episode_soc_violations')
    axes[0, 2].plot(soc_series)
    axes[0, 2].set_title('SoC Violations per Episode (Realized)')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Violations')
    axes[0, 2].grid(True)
    
    # Evaluation rewards
    if metrics['eval_rewards']:
        eval_episodes = np.arange(0, len(metrics['episode_rewards']), config['training']['eval_every'])
        axes[1, 0].plot(eval_episodes, metrics['eval_rewards'], 'ro-')
        axes[1, 0].set_title('Evaluation Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True)
    
    # Average actions
    axes[1, 1].plot(metrics['episode_actions'], label='|Action| (legacy avg)', color='gray', alpha=0.5)
    if 'episode_actions_raw' in metrics:
        axes[1, 1].plot(metrics['episode_actions_raw'], label='|a_raw|', color='orange')
    if 'episode_actions_safe' in metrics:
        axes[1, 1].plot(metrics['episode_actions_safe'], label='|a_safe|', color='blue')
    axes[1, 1].set_title('Average Action Magnitude (Raw vs Safe)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('|Action|')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    # Revenue vs Cost
    if metrics['episode_revenues'] and metrics['episode_costs']:
        axes[1, 2].plot(metrics['episode_revenues'], label='Revenue', color='green')
        axes[1, 2].plot(metrics['episode_costs'], label='Cost', color='red')
        axes[1, 2].set_title('Episode Revenue vs Cost')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Amount ($)')
        axes[1, 2].grid(True)
        axes[1, 2].legend()
    
    # SoC trajectory example (last episode)
    if metrics['episode_soc_trajectories']:
        last_soc = metrics['episode_soc_trajectories'][-1]
        steps = range(len(last_soc))
        axes[2, 0].plot(steps, last_soc, 'b-', linewidth=2)
        axes[2, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='SoC Min')
        axes[2, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='SoC Max')
        axes[2, 0].set_title('SoC Trajectory (Last Episode)')
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('SoC')
        axes[2, 0].grid(True)
        axes[2, 0].legend()
    
    # 改為顯示三個KPI：Attempted / Projected / Realized（每集總次數）
    num_eps = len(metrics.get('episode_rewards', []))
    def pad_series(key: str):
        arr = metrics.get(key, []) or []
        if num_eps <= 0:
            return []
        if len(arr) < num_eps:
            arr = list(arr) + [0] * (num_eps - len(arr))
        elif len(arr) > num_eps:
            arr = list(arr[:num_eps])
        return arr
    att_series = pad_series('episode_attempted_violations')
    proj_series = pad_series('episode_safety_projected')
    real_series = pad_series('episode_realized_violations') or pad_series('episode_soc_violations')
    if num_eps > 0:
        att = np.array(att_series, dtype=float)
        proj = np.array(proj_series, dtype=float)
        real = np.array(real_series, dtype=float)
        steps = float(config['training'].get('max_steps', len(real)))
        overlap_score = (np.mean(np.abs(att - proj)) + np.mean(np.abs(proj - real))) / max(1.0, steps)
        if overlap_score < 0.02:
            # 幾乎重疊：改用堆疊長條，分解三部分
            no_intervention = np.clip(att - proj, 0, None)
            prevented = np.clip(proj - real, 0, None)
            failed = np.clip(real, 0, None)
            x = np.arange(num_eps)
            axes[2, 1].bar(x, no_intervention, label='Attempted w/o SN', color='gray', alpha=0.5)
            axes[2, 1].bar(x, prevented, bottom=no_intervention, label='SN Prevented', color='tab:blue', alpha=0.7)
            axes[2, 1].bar(x, failed, bottom=no_intervention+prevented, label='Realized', color='tab:red', alpha=0.8)
            axes[2, 1].set_title('SafetyNet Outcomes per Episode (Stacked)')
        else:
            # 差異明顯：畫三條折線 + 標記
            axes[2, 1].plot(att_series, label='Attempted', color='orange', marker='o', markevery=max(1, num_eps//10), alpha=0.7)
            axes[2, 1].plot(proj_series, label='Projected', color='blue', marker='s', markevery=max(1, num_eps//10), alpha=0.7)
            axes[2, 1].plot(real_series, label='Realized', color='red', marker='x', markevery=max(1, num_eps//10), alpha=0.8)
            # 額外畫差分線幫助分辨
            d1 = (att - proj).tolist()
            d2 = (proj - real).tolist()
            axes[2, 1].plot(d1, label='Δ(Att-Proj)', color='tab:gray', linestyle='--', alpha=0.4)
            axes[2, 1].plot(d2, label='Δ(Proj-Real)', color='tab:green', linestyle='--', alpha=0.4)
            axes[2, 1].set_title('SafetyNet KPIs per Episode (Lines + Deltas)')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Count (<= steps)')
        axes[2, 1].grid(True)
        axes[2, 1].legend()

    # Net profit (Revenue - Cost)
    if metrics['episode_revenues'] and metrics['episode_costs']:
        net_profits = [r - c for r, c in zip(metrics['episode_revenues'], metrics['episode_costs'])]
        axes[2, 2].plot(net_profits, color='purple')
        axes[2, 2].set_title('Net Profit per Episode')
        axes[2, 2].set_xlabel('Episode')
        axes[2, 2].set_ylabel('Net Profit ($)')
        axes[2, 2].grid(True)
        axes[2, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save to experiment directory
    exp_plot_path = os.path.join(exp_manager.results_dir, "training_results.png")
    plt.savefig(exp_plot_path, dpi=300, bbox_inches='tight')
    print(f"Training plot saved to: {exp_plot_path}")
    
    # Also save locally for display
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main training function with microgrid environment"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train SAC agent with Microgrid Environment')
    parser.add_argument('--config', type=str, default='../configs/config_microgrid.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Override total episodes from config')
    parser.add_argument('--name', type=str, default=None,
                       help='Custom experiment name')
    parser.add_argument('--variant', type=str, choices=['sac', 'sac_sn', 'sac_sn_evi'], default=None,
                       help='Training variant: sac (baseline), sac_sn (with SafetyNet), sac_sn_evi (with SafetyNet + Evidential)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    # Setup conformal parameters if provided
    conformal_cfg = config.get('conformal', {})
    try:
        set_conformal_params(
            window=int(conformal_cfg.get('window', 2880)),
            delta=float(conformal_cfg.get('delta', 0.1))
        )
    except Exception:
        set_conformal_params()
    
    # Override episodes if specified
    if args.episodes is not None:
        config['training']['total_episodes'] = args.episodes
    # Apply variant override if provided
    if args.variant is not None:
        config.setdefault('training', {})['variant'] = args.variant
    
    # Create experiment manager
    exp_manager = create_experiment_from_config(args.config, args.name)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Create microgrid environment
    env = create_environment(config)
    
    # Agent parameters
    state_dim = env.observation_space.shape[0]  # 6D state space
    action_dim = env.action_space.shape[0]      # 1D action space
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Device
    device_config = config['device']
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    print(f"Using device: {device}")
    
    # Create SAC agent
    agent = create_agent(config, state_dim, action_dim, device)
    
    # Train agent
    start_time = time.time()
    metrics = train_sac_with_microgrid(env, agent, config, exp_manager)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    if metrics['eval_rewards']:
        print(f"Final evaluation reward: {metrics['eval_rewards'][-1]:.2f}")
    print(f"Average SoC violations: {np.mean(metrics['episode_soc_violations']):.2f}")
    print(f"Average action violations: {np.mean(metrics['episode_action_violations']):.2f}")
    
    # Collect compute resources information
    compute_res = collect_compute_resources(agent, device, training_time)
    
    # Save results to experiment directory
    metadata = {
        'variant': config.get('training', {}).get('variant', 'sac'),
        'time_step': float(getattr(env, 'time_step', 1.0)),
        'seed': int(config.get('random_seed', 0)),
    }
    exp_manager.save_results(metrics, metadata=metadata, compute_resources=compute_res)
    
    # Print compute resources summary
    print("\n" + "="*50)
    print(format_compute_resources(compute_res))
    print("="*50)
    
    # Plot results
    if config['logging']['plot_results']:
        plot_microgrid_training_results(metrics, config, exp_manager)
    
    # Save metrics
    if config['logging']['save_metrics']:
        metrics_path = "sac_training_metrics.npz"
        np.savez(metrics_path, **metrics)
        exp_manager.save_metrics(metrics_path)
        print("Training metrics saved to experiment directory")
    
    # Print final statistics
    print("\nFinal Training Statistics:")
    print(f"Total episodes: {len(metrics['episode_rewards'])}")
    print(f"Average reward: {np.mean(metrics['episode_rewards']):.2f}")
    print(f"Best reward: {np.max(metrics['episode_rewards']):.2f}")
    print(f"Average SoC violations: {np.mean(metrics['episode_soc_violations']):.2f}")
    print(f"Total SoC violations: {np.sum(metrics['episode_soc_violations'])}")
    print(f"Average revenue: ${np.mean(metrics['episode_revenues']):.2f}")
    print(f"Average cost: ${np.mean(metrics['episode_costs']):.2f}")
    print(f"Net profit: ${np.mean(metrics['episode_revenues']) - np.mean(metrics['episode_costs']):.2f}")
    
    # Print experiment summary
    exp_manager.print_experiment_summary()
    
    # Clean up temporary files
    exp_manager.cleanup_temp_files()


if __name__ == "__main__":
    main() 