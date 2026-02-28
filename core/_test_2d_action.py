"""Quick test: 2D action space (power + flow_rate) with sac_sn variant."""
import yaml, sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

with open(os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_p302_sim.yaml'), 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

config['training']['total_episodes'] = 3
config['training']['eval_every'] = 2
config['sac']['warmup_steps'] = 100

from train_sac_microgrid import create_environment, create_agent
import torch

env = create_environment(config)
print(f'Action space: {env.action_space}')
print(f'Observation space: {env.observation_space}')
print(f'use_flow_rate_action: {env.use_flow_rate_action}')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print(f'state_dim={state_dim}, action_dim={action_dim}')

device = 'cpu'
agent = create_agent(config, state_dim, action_dim, device)
print(f'Agent created: actor output dim = {agent.actor.fc_mean.out_features}')

# Quick step test
state, info = env.reset()
print(f'Initial state: {state}')
action = agent.select_action(state, evaluate=False)
print(f'Action (norm): {action}')
print(f'Action shape: {action.shape}')

power_norm = float(action[0])
flow_norm_raw = float(action[1])
power_kw = power_norm * env.battery_power_kw
flow_frac = float(np.clip((flow_norm_raw + 1.0) * 0.495 + 0.01, 0.01, 1.0))
print(f'power_kw={power_kw:.6f}, flow_fraction={flow_frac:.3f}')

next_state, reward, term, trunc, step_info = env.step([power_kw, flow_frac])
print(f'Reward: {reward:.4f}, SoC: {next_state[0]:.4f}')
print(f'Step info keys: {list(step_info.keys())}')
print(f'flow_rate_lpm: {step_info.get("flow_rate_lpm", "N/A")}')
print(f'pump_power_kw: {step_info.get("pump_power_kw", "N/A")}')
print(f'flow_efficiency: {step_info.get("flow_efficiency", "N/A")}')
print('=== 2D Action Space Test PASSED ===')

