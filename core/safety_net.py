import numpy as np
from typing import Tuple, Dict, Any, Optional
import warnings

import math
from typing import Tuple, Optional
from collections import deque


warnings.filterwarnings('ignore')

_conformal_window = 2880
_conformal_delta = 0.1
_residual_buffer: deque[float] = deque(maxlen=_conformal_window)


def set_conformal_params(window: int = 2880, delta: float = 0.1):
    global _conformal_window, _conformal_delta, _residual_buffer
    _conformal_window = max(10, int(window))
    _conformal_delta = float(np.clip(delta, 1e-4, 0.5))
    # 重新建立 buffer 以適配新窗口
    old_vals = list(_residual_buffer) if _residual_buffer else []
    _residual_buffer = deque(old_vals[-_conformal_window:], maxlen=_conformal_window)


def update_conformal_residual(residual: float):
    try:
        r = float(abs(residual))
    except Exception:
        return
    _residual_buffer.append(r)


def clear_residual_buffer():
    """清空 Conformal 殘差緩衝（保留窗口大小）。"""
    global _residual_buffer
    _residual_buffer = deque(maxlen=_conformal_window)


def get_residual_count() -> int:
    """取得目前殘差樣本數。"""
    try:
        return int(len(_residual_buffer))
    except Exception:
        return 0


def _conformal_tube() -> float:
    if not _residual_buffer:
        return 0.0
    arr = np.asarray(_residual_buffer, dtype=float)
    q = min(0.999, max(0.0, 1.0 - _conformal_delta))
    try:
        return float(np.quantile(arr, q))
    except Exception:
        return float(np.max(arr))

class SafetyNet:
    """
    SafetyNet: 動作投影/屏蔽系統 - Phase-1 增強版
    
    功能：
    - 計算安全動作邊界
    - 投影不安全動作到安全域
    - 提供動態安全緩衝區管理
    - 支持 N-step 前視約束
    - 邊界內縮避免誤判
    """
    
    def __init__(
        self,
        battery_capacity_kwh: float = 100.0,
        battery_power_kw: float = 50.0,
        battery_efficiency: float = 0.95,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        initial_buffer_ratio: float = 0.05,  # 起始緩衝區
        min_buffer_ratio: float = 0.02,      # 最小緩衝區
        buffer_decay_episodes: int = 10,     # 收縮觸發條件
        buffer_decay_rate: float = 0.01,     # 收縮速率
        boundary_epsilon: float = 0.005,     # 邊界內縮
        time_step: float = 1.0,              # 小時
        n_step_preview: int = 2,             # N步前視
        enable_n_step_preview: bool = True
    ):
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_power_kw = battery_power_kw
        self.battery_efficiency = battery_efficiency
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.initial_buffer_ratio = initial_buffer_ratio
        self.min_buffer_ratio = min_buffer_ratio
        self.buffer_decay_episodes = buffer_decay_episodes
        self.buffer_decay_rate = buffer_decay_rate
        self.boundary_epsilon = boundary_epsilon
        self.time_step = time_step
        self.n_step_preview = n_step_preview
        self.enable_n_step_preview = enable_n_step_preview
        
        # 動態緩衝區管理
        self.current_buffer_ratio = initial_buffer_ratio
        self.consecutive_safe_episodes = 0
        
        # 計算當前安全緩衝區
        self._update_safe_bounds()
        
        print(f"SafetyNet Phase-1 initialized:")
        print(f"  - Battery capacity: {battery_capacity_kwh:.1f} kWh")
        print(f"  - Battery power: {battery_power_kw:.1f} kW")
        print(f"  - Initial safe SoC range: [{self.safe_soc_min:.3f}, {self.safe_soc_max:.3f}]")
        print(f"  - Buffer ratio: {self.current_buffer_ratio:.1%}")
        print(f"  - N-step preview: {n_step_preview}")
        print(f"  - Boundary epsilon: {boundary_epsilon:.3f}")
    
    def _update_safe_bounds(self):
        """更新安全邊界（考慮當前緩衝區和內縮）"""
        buffer_size = (self.soc_max - self.soc_min) * self.current_buffer_ratio
        self.safe_soc_min = self.soc_min + buffer_size + self.boundary_epsilon
        self.safe_soc_max = self.soc_max - buffer_size - self.boundary_epsilon
    
    def _calculate_soc_bounds(self, current_soc: float) -> Tuple[float, float]:
        """
        計算基於當前SoC的安全動作邊界
        
        Args:
            current_soc: 當前電池SoC
            
        Returns:
            (min_power, max_power): 安全功率範圍 (kW)
        """
        # 基礎功率限制
        base_min_power = -self.battery_power_kw
        base_max_power = self.battery_power_kw
        
        # 檢查是否在安全緩衝區內
        in_buffer_zone = (current_soc <= self.safe_soc_min or 
                         current_soc >= self.safe_soc_max)
        
        if in_buffer_zone:
            # 在緩衝區內，縮小允許的功率範圍
            base_min_power *= 0.5  # 使用固定縮放係數
            base_max_power *= 0.5
        
        return base_min_power, base_max_power
    
    def _calculate_preview_bounds(self, current_soc: float) -> Tuple[float, float]:
        """
        根據 SoC 與效率/時間步長，直接計算單步可行的最大充放電功率範圍，
        使用 safe 邊界以保守化，並可按 n_step_preview 收斂動作幅度。
        """
        dt = max(float(self.time_step), 1e-9)
        eta = max(float(self.battery_efficiency), 1e-9)
        cap = max(float(self.battery_capacity_kwh), 1e-9)
        # 使用保守邊界（safe）
        soc_min_eff = float(getattr(self, 'safe_soc_min', self.soc_min))
        soc_max_eff = float(getattr(self, 'safe_soc_max', self.soc_max))
        # 剩餘可充/放能量（kWh）
        charge_room_kwh = max(0.0, (soc_max_eff - current_soc) * cap)
        discharge_room_kwh = max(0.0, (current_soc - soc_min_eff) * cap)
        # 轉換為單步功率限制（kW）
        max_charge_kw = charge_room_kwh / (dt * eta)  # 充電需除以效率
        max_discharge_kw = discharge_room_kwh * eta / dt  # 放電乘以效率
        # 按 n-step 前視收斂（保守化）
        if self.enable_n_step_preview and int(self.n_step_preview) > 1:
            factor = 1.0 / float(int(self.n_step_preview))
            max_charge_kw *= factor
            max_discharge_kw *= factor
        # 限制在物理功率範圍內
        max_charge_kw = min(max_charge_kw, self.battery_power_kw)
        max_discharge_kw = min(max_discharge_kw, self.battery_power_kw)
        return -max_discharge_kw, max_charge_kw
    
    def bounds(self, state: np.ndarray) -> Tuple[float, float]:
        """
        計算給定狀態下的安全動作邊界
        
        Args:
            state: 環境狀態 [SoC, load, pv, price, hour, day]
            
        Returns:
            (a_low, a_high): 安全動作範圍
        """
        current_soc = state[0]
        
        # 基礎邊界
        base_min, base_max = self._calculate_soc_bounds(current_soc)
        
        # 前視約束邊界（不依賴特定動作，直接給出可行功率範圍）
        preview_min, preview_max = self._calculate_preview_bounds(current_soc)
        
        # 取交集
        a_low = max(base_min, preview_min)
        a_high = min(base_max, preview_max)
        
        # 確保邊界有效
        a_low = max(a_low, -self.battery_power_kw)
        a_high = min(a_high, self.battery_power_kw)
        
        return a_low, a_high
    
    def project(self, state: np.ndarray, action_raw: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        將原始動作投影到安全域
        
        Args:
            state: 環境狀態
            action_raw: 原始動作
            
        Returns:
            (action_safe, info): 安全動作和投影信息
        """
        action_value = float(action_raw[0])
        
        # 計算安全邊界
        a_low, a_high = self.bounds(state)
        
        # 投影動作
        action_safe = np.clip(action_value, a_low, a_high)
        
        # 檢查是否被裁剪
        clipped = abs(action_safe - action_value) > 1e-6
        
        # 計算裁剪程度
        if clipped:
            if action_value < a_low:
                clip_ratio = (a_low - action_value) / abs(action_value) if action_value != 0 else 1.0
            else:
                clip_ratio = (action_value - a_high) / abs(action_value) if action_value != 0 else 1.0
        else:
            clip_ratio = 0.0
        
        # 準備信息
        info = {
            'clipped': clipped,
            'clip_ratio': clip_ratio,
            'original_action': action_value,
            'safe_bounds': (a_low, a_high),
            'soc': state[0],
            'in_buffer_zone': (state[0] <= self.safe_soc_min or state[0] >= self.safe_soc_max)
        }
        
        return np.array([action_safe], dtype=np.float32), info
    
    def get_safety_metrics(self, state: np.ndarray) -> Dict[str, Any]:
        """
        獲取安全指標
        
        Args:
            state: 環境狀態
            
        Returns:
            安全指標字典
        """
        current_soc = state[0]
        
        # 計算到邊界的距離
        distance_to_min = current_soc - self.soc_min
        distance_to_max = self.soc_max - current_soc
        
        # 計算到安全邊界的距離
        distance_to_safe_min = current_soc - self.safe_soc_min
        distance_to_safe_max = self.safe_soc_max - current_soc
        
        # 判斷安全等級
        buffer_size = (self.soc_max - self.soc_min) * self.current_buffer_ratio
        if current_soc < self.safe_soc_min:
            safety_level = "danger_low"
        elif current_soc > self.safe_soc_max:
            safety_level = "danger_high"
        elif current_soc < self.safe_soc_min + buffer_size * 0.5:
            safety_level = "warning_low"
        elif current_soc > self.safe_soc_max - buffer_size * 0.5:
            safety_level = "warning_high"
        else:
            safety_level = "safe"
        
        return {
            'safety_level': safety_level,
            'distance_to_min': distance_to_min,
            'distance_to_max': distance_to_max,
            'distance_to_safe_min': distance_to_safe_min,
            'distance_to_safe_max': distance_to_safe_max,
            'buffer_utilization': 1.0 - min(distance_to_safe_min, distance_to_safe_max) / buffer_size,
            'current_soc': current_soc,
            'safe_range': (self.safe_soc_min, self.safe_soc_max)
        }
    
    def update_parameters(self, **kwargs):
        """
        動態更新SafetyNet參數
        
        Args:
            **kwargs: 要更新的參數
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Updated {key} to {value}")
        
        # 重新計算安全緩衝區
        if 'buffer_ratio' in kwargs or 'soc_min' in kwargs or 'soc_max' in kwargs:
            self._update_safe_bounds()
            print(f"Recalculated safe SoC range: [{self.safe_soc_min:.3f}, {self.safe_soc_max:.3f}]")

    def update_buffer_after_episode(self, realized_violations: int):
        """根據 episode 結果更新動態緩衝區"""
        if realized_violations == 0:
            self.consecutive_safe_episodes += 1
            if self.consecutive_safe_episodes >= self.buffer_decay_episodes:
                # 收縮緩衝區
                new_buffer = self.current_buffer_ratio - self.buffer_decay_rate
                if new_buffer >= self.min_buffer_ratio:
                    self.current_buffer_ratio = new_buffer
                    self._update_safe_bounds()
                    print(f"SafetyNet: Buffer shrunk to {self.current_buffer_ratio:.1%}")
        else:
            # 重置安全計數
            self.consecutive_safe_episodes = 0
    
    def _n_step_soc_prediction(self, current_soc: float, action: float, env) -> float:
        """N-step SoC 預測"""
        if not self.enable_n_step_preview:
            return current_soc
        
        soc = current_soc
        for step in range(self.n_step_preview):
            # 使用環境的預測方法
            if hasattr(env, 'predict_soc_raw'):
                soc = env.predict_soc_raw(soc, action)
            else:
                # 備用預測
                energy_change_kwh = action * self.time_step
                if energy_change_kwh > 0:
                    energy_change_kwh *= self.battery_efficiency
                else:
                    energy_change_kwh /= self.battery_efficiency
                soc += energy_change_kwh / self.battery_capacity_kwh
        
        return soc


class SafetyNetWrapper:
    """
    SafetyNet的包裝器，提供統一的接口
    """
    
    def __init__(self, safety_net: SafetyNet):
        self.safety_net = safety_net
    
    def __call__(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """調用SafetyNet進行動作投影"""
        return self.safety_net.project(state, action)
    
    def bounds(self, state: np.ndarray) -> Tuple[float, float]:
        """獲取安全邊界"""
        return self.safety_net.bounds(state)
    
    def get_safety_metrics(self, state: np.ndarray) -> Dict[str, Any]:
        """獲取安全指標"""
        return self.safety_net.get_safety_metrics(state)


def create_safety_net(
    battery_capacity_kwh: float = 100.0,
    battery_power_kw: float = 50.0,
    battery_efficiency: float = 0.95,
    soc_min: float = 0.1,
    soc_max: float = 0.9,
    buffer_ratio: float = 0.08,
    buffer_scale_kappa: float = 0.5,
    enable_one_step_preview: bool = True
) -> SafetyNet:
    """
    創建SafetyNet的工廠函數
    
    Args:
        battery_capacity_kwh: 電池容量 (kWh)
        battery_power_kw: 電池功率 (kW)
        battery_efficiency: 電池效率
        soc_min: 最小SoC
        soc_max: 最大SoC
        buffer_ratio: 安全緩衝區比例
        buffer_scale_kappa: 邊界縮放係數
        enable_one_step_preview: 是否啟用一階前視
        
    Returns:
        SafetyNet: 配置好的安全網
    """
    return SafetyNet(
        battery_capacity_kwh=battery_capacity_kwh,
        battery_power_kw=battery_power_kw,
        battery_efficiency=battery_efficiency,
        soc_min=soc_min,
        soc_max=soc_max,
        buffer_ratio=buffer_ratio,
        buffer_scale_kappa=buffer_scale_kappa,
        enable_one_step_preview=enable_one_step_preview
    ) 


def _clip(value: float, low: float, high: float) -> float:
	return max(low, min(high, value))


def _apply_ramp_limit(a_new: float, a_prev: float, ramp_kw: Optional[float]) -> float:
	if ramp_kw is None or ramp_kw <= 0:
		return a_new
	return _clip(a_new, a_prev - ramp_kw, a_prev + ramp_kw)


def _solve_a_for_soc_target(soc: float, soc_target: float, env) -> float:
	"""根據目標 SoC 反解動作 a（kW）。
	E = (soc_target - soc) * cap
	若 E >= 0（充電）：a = E / (dt * eta)
	若 E < 0（放電）：a = E * eta / dt
	"""
	dt = float(getattr(env, 'time_step', 1.0))
	eta = float(getattr(env, 'battery_efficiency', 1.0))
	cap = float(getattr(env, 'battery_capacity_kwh', 1.0))
	if cap <= 0 or dt <= 0 or eta <= 0:
		return 0.0
	E = (soc_target - soc) * cap
	if E >= 0:
		return E / (dt * eta)
	else:
		return E * eta / dt


def _project_soc_safe(soc: float, a: float, soc_min: float, soc_max: float, env) -> float:
	"""若以 a 前進會越界，將 a 調整到剛好到邊界；否則回傳 a。"""
	predict = getattr(env, 'predict_soc_raw', None)
	if predict is None:
		return a
	soc_next = float(predict(soc, a))
	if soc_min <= soc_next <= soc_max:
		return a
	# 選擇靠近的邊界作為目標，避免跨越另一側
	target = soc_min if soc_next < soc_min else soc_max
	return _solve_a_for_soc_target(soc, target, env)



def project(
	state: np.ndarray,
	action: np.ndarray,
	prev_action: float,
	pmax: float,
	ramp_kw: Optional[float],
	soc_bounds: Tuple[float, float],
	env,
) -> Tuple[float, bool, float]:
	"""SafetyNet 投影：Pmax → ramp → SoC 投影 → 再 Pmax → 最終驗證與縮放。
	回傳 (a_safe, changed, delta_kw) 其中 delta_kw = |a_safe - a_raw|。
	"""
	a_raw = float(action[0])
	soc = float(state[0])
	soc_min, soc_max = soc_bounds
	# Conformal risk-tube：以殘差分位數縮邊界
	tube = _conformal_tube()
	soc_min_eff = float(soc_min + tube)
	soc_max_eff = float(soc_max - tube)
	changed = False
	# 1) Pmax
	a1 = _clip(a_raw, -abs(pmax), abs(pmax))
	changed = changed or (abs(a1 - a_raw) > 1e-8)
	# 2) ramp（若提供）
	a2 = _apply_ramp_limit(a1, float(prev_action), ramp_kw) if ramp_kw is not None else a1
	# 2) SoC 邊界投影
	a3 = _project_soc_safe(soc, a2, soc_min_eff, soc_max_eff, env)
	changed = changed or (abs(a3 - a2) > 1e-8)
	# 3) 再次 Pmax 校正
	a4 = _clip(a3, -abs(pmax), abs(pmax))
	if abs(a4 - a3) > 1e-8:
		changed = True
	# 4) 最終驗證：若仍將越界，按比例縮放到安全區間
	predict = getattr(env, 'predict_soc_raw', None)
	if predict is not None:
		soc_next = float(predict(soc, a4))
		if soc_next < soc_min_eff or soc_next > soc_max_eff:
			low, high = 0.0, 1.0
			base = a4
			for _ in range(12):
				mid = (low + high) * 0.5
				a_try = base * mid
				soc_try = float(predict(soc, a_try))
				if soc_min_eff <= soc_try <= soc_max_eff:
					low = mid
				else:
					high = mid
			a4 = base * low
			changed = True
	delta_kw = abs(float(a4) - a_raw)
	return float(a4), changed, float(delta_kw)