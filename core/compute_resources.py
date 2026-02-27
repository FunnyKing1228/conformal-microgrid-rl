"""
計算資源監控工具
用於記錄實驗的計算資源使用情況（類似LLM研究中的算力指標）
"""
import torch
import platform
from typing import Dict, Any, Optional
import psutil


def get_device_info(device: str) -> Dict[str, Any]:
    info = {
        'device_type': device,
        'gpu_name': None,
        'gpu_memory_total_gb': None,
        'cpu_name': platform.processor(),
        'cpu_cores': psutil.cpu_count(logical=False),
        'cpu_threads': psutil.cpu_count(logical=True),
        'system_memory_total_gb': psutil.virtual_memory().total / (1024**3)
    }
    if device.startswith('cuda') and torch.cuda.is_available():
        try:
            gpu_id = 0
            if ':' in device:
                gpu_id = int(device.split(':')[1])
            info['gpu_name'] = torch.cuda.get_device_name(gpu_id)
            info['gpu_memory_total_gb'] = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            info['gpu_compute_capability'] = f"{torch.cuda.get_device_capability(gpu_id)[0]}.{torch.cuda.get_device_capability(gpu_id)[1]}"
        except Exception as e:
            print(f"Warning: Could not get GPU info: {e}")
    return info


def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'non_trainable_parameters': int(total_params - trainable_params)
    }


def count_agent_parameters(agent: Any) -> Dict[str, Any]:
    params = {}
    if hasattr(agent, 'actor'):
        params['actor'] = count_model_parameters(agent.actor)
    if hasattr(agent, 'critic1'):
        params['critic1'] = count_model_parameters(agent.critic1)
    if hasattr(agent, 'critic2'):
        params['critic2'] = count_model_parameters(agent.critic2)
    if hasattr(agent, 'target_critic1'):
        params['target_critic1'] = count_model_parameters(agent.target_critic1)
    if hasattr(agent, 'target_critic2'):
        params['target_critic2'] = count_model_parameters(agent.target_critic2)
    if hasattr(agent, 'occ_head'):
        params['occ_head'] = count_model_parameters(agent.occ_head)
    total_all = 0
    trainable_all = 0
    for net_name, net_params in params.items():
        if isinstance(net_params, dict):
            total_all += net_params.get('total_parameters', 0)
            trainable_all += net_params.get('trainable_parameters', 0)
    params['total_all_networks'] = {
        'total_parameters': total_all,
        'trainable_parameters': trainable_all
    }
    return params


def calculate_compute_hours(duration_seconds: float, device: str) -> Dict[str, float]:
    hours = duration_seconds / 3600.0
    if device.startswith('cuda'):
        return {'gpu_hours': hours, 'cpu_hours': 0.0}
    else:
        return {'gpu_hours': 0.0, 'cpu_hours': hours}


def collect_compute_resources(agent: Any, device: str, duration_seconds: float) -> Dict[str, Any]:
    resources = {
        'device_info': get_device_info(device),
        'compute_hours': calculate_compute_hours(duration_seconds, device),
        'duration_seconds': float(duration_seconds)
    }
    if agent is not None:
        try:
            resources['model_parameters'] = count_agent_parameters(agent)
        except Exception as e:
            print(f"Warning: Could not count agent parameters: {e}")
            resources['model_parameters'] = None
    else:
        resources['model_parameters'] = None
    return resources


def format_compute_resources(resources: Dict[str, Any]) -> str:
    lines = []
    lines.append("計算資源使用情況 (Compute Resources):")
    lines.append("-" * 50)
    if 'device_info' in resources:
        di = resources['device_info']
        lines.append(f"設備類型: {di.get('device_type', 'unknown')}")
        if di.get('gpu_name'):
            lines.append(f"GPU: {di['gpu_name']}")
            lines.append(f"GPU記憶體: {di.get('gpu_memory_total_gb', 0):.2f} GB")
        lines.append(f"CPU: {di.get('cpu_name', 'unknown')}")
        lines.append(f"CPU核心數: {di.get('cpu_cores', 'unknown')}")
        lines.append(f"系統記憶體: {di.get('system_memory_total_gb', 0):.2f} GB")
    if 'duration_seconds' in resources:
        duration = resources['duration_seconds']
        hours = duration / 3600.0
        lines.append(f"\n訓練時間: {hours:.2f} 小時 ({duration:.1f} 秒)")
    if 'compute_hours' in resources:
        ch = resources['compute_hours']
        if ch.get('gpu_hours', 0) > 0:
            lines.append(f"GPU小時: {ch['gpu_hours']:.4f}")
        if ch.get('cpu_hours', 0) > 0:
            lines.append(f"CPU小時: {ch['cpu_hours']:.4f}")
    if 'model_parameters' in resources and resources['model_parameters']:
        mp = resources['model_parameters']
        if 'total_all_networks' in mp:
            total = mp['total_all_networks']
            lines.append(f"\n模型參數總數: {total.get('total_parameters', 0):,}")
            lines.append(f"可訓練參數: {total.get('trainable_parameters', 0):,}")
    return "\n".join(lines)
