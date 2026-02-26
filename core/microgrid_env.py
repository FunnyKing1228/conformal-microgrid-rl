import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from pymgrid import MicrogridGenerator
    from pymgrid.microgrid import Microgrid
    PYTHON_MICROGRID_AVAILABLE = True
except ImportError:
    PYTHON_MICROGRID_AVAILABLE = False
    print("Info: python-microgrid not installed. Running MicrogridEnvironment in python-microgrid synthetic mode.")


class MicrogridEnvironment(gym.Env):
    """
    基於 python-microgrid 介面的微電網環境
    - 若系統未安裝 python-microgrid，則以『合成資料的 python-microgrid 模式』運行（行為一致、資料源合成）
    """
    
    def __init__(
        self,
        microgrid_id: int = 0,
        episode_length: int = 24,
        time_step: int = 1,  # hours
        battery_capacity_kwh: float = 100.0,
        battery_power_kw: float = 50.0,
        battery_efficiency: float = 0.95,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        price_scaling: float = 1.0,
        reward_scaling: float = 1.0,
        use_real_data: bool = True,
        ramp_limit_kw: float = None,  # 新增：爬坡限制
        hard_guard: bool = False,  # Phase-2: 硬護欄（訓練關閉、評估開啟）
        allow_grid_trading: bool = True,  # 是否允許與電網買賣電（False = 孤島模式）
        # External dataset (CSV) options
        dataset_csv_path: Optional[str] = None,
        dataset_pv_join_wind: bool = False,
        train_window_hours: Optional[int] = None,
        dataset_pv_column: Optional[str] = None,
        dataset_load_kw: Optional[float] = None,
        dataset_power_scale: float = 1.0,
        dataset_time_column: Optional[str] = None,
        # Weather variation (synthetic data only)
        weather_pv_scale_std: float = 0.0,   # 每回合 PV 乘性變化（雲量）
        weather_load_scale_std: float = 0.0, # 每回合負載乘性變化
        weather_pv_noise_std: float = 0.0,   # PV 每步隨機波動（比例）
        # Synthetic hourly-hold pattern (for 15-min actions)
        synthetic_hourly_hold: bool = False,
        synthetic_pv_peak_kw: float = 20.0,
        synthetic_pv_start_hour: int = 6,
        synthetic_pv_end_hour: int = 18,
        synthetic_load_base_kw: float = 10.0,
        synthetic_load_amp_kw: float = 5.0,
        synthetic_price_base: float = 0.12,
        synthetic_price_peak: float = 0.20,
        synthetic_price_peak_start: int = 8,
        synthetic_price_peak_end: int = 18,
        # Stress knobs（預設關閉）
        stress_enable: bool = False,
        stress_efficiency_noise_std: float = 0.0,
        stress_dt_jitter_std: float = 0.0,
        stress_action_lag_alpha: float = 0.0,
        stress_soc_obs_delay: int = 0,
        stress_soc_obs_noise_std: float = 0.0,
        stress_bounds_drift_std: float = 0.0,
        stress_external_pmax_shrink_prob: float = 0.0,
        stress_external_pmax_shrink_factor: float = 1.0,
        stress_power_loss_ratio: float = 0.0  # 功率損耗比例（0-1），充電時增加需求，放電時減少輸出
    ):
        super().__init__()
        
        self.episode_length = episode_length
        self.time_step = time_step
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_power_kw = battery_power_kw
        self.battery_efficiency = battery_efficiency
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.price_scaling = price_scaling
        self.reward_scaling = reward_scaling
        self.use_real_data = use_real_data
        self.ramp_limit_kw = ramp_limit_kw  # 新增
        self.hard_guard = hard_guard
        self.allow_grid_trading = bool(allow_grid_trading)  # 是否允許電網交易
        self.dataset_csv_path = dataset_csv_path
        self.dataset_pv_join_wind = bool(dataset_pv_join_wind)
        self.train_window_hours = int(train_window_hours) if train_window_hours is not None else None
        self.dataset_pv_column = dataset_pv_column
        self.dataset_load_kw = float(dataset_load_kw) if dataset_load_kw is not None else None
        self.dataset_power_scale = float(dataset_power_scale)
        self.dataset_time_column = dataset_time_column
        # Weather variation settings
        self.weather_pv_scale_std = float(weather_pv_scale_std)
        self.weather_load_scale_std = float(weather_load_scale_std)
        self.weather_pv_noise_std = float(weather_pv_noise_std)
        self.synthetic_hourly_hold = bool(synthetic_hourly_hold)
        self.synthetic_pv_peak_kw = float(synthetic_pv_peak_kw)
        self.synthetic_pv_start_hour = int(synthetic_pv_start_hour)
        self.synthetic_pv_end_hour = int(synthetic_pv_end_hour)
        self.synthetic_load_base_kw = float(synthetic_load_base_kw)
        self.synthetic_load_amp_kw = float(synthetic_load_amp_kw)
        self.synthetic_price_base = float(synthetic_price_base)
        self.synthetic_price_peak = float(synthetic_price_peak)
        self.synthetic_price_peak_start = int(synthetic_price_peak_start)
        self.synthetic_price_peak_end = int(synthetic_price_peak_end)
        # 固定起點（供評估鎖定驗證窗使用）；None 表示隨機
        self.fixed_start_idx: Optional[int] = None
        # Stress settings
        self.stress_enable = bool(stress_enable)
        self.stress_efficiency_noise_std = float(stress_efficiency_noise_std)
        self.stress_dt_jitter_std = float(stress_dt_jitter_std)
        self.stress_action_lag_alpha = float(stress_action_lag_alpha)
        self.stress_soc_obs_delay = int(stress_soc_obs_delay)
        self.stress_soc_obs_noise_std = float(stress_soc_obs_noise_std)
        self.stress_bounds_drift_std = float(stress_bounds_drift_std)
        self.stress_external_pmax_shrink_prob = float(stress_external_pmax_shrink_prob)
        self.stress_external_pmax_shrink_factor = float(stress_external_pmax_shrink_factor)
        self.stress_power_loss_ratio = float(np.clip(stress_power_loss_ratio, 0.0, 0.3))  # 限制在0-30%
        # Effective parameters for stress
        self.soc_min_eff = self.soc_min
        self.soc_max_eff = self.soc_max
        self._effective_time_step = float(self.time_step)
        self._effective_efficiency = float(self.battery_efficiency)
        self._prev_exec_action_kw = 0.0
        self._soc_obs_buffer = []
        # 允許從外部調整的懲罰（若訓練程式未注入，維持預設）
        self.realized_violation_penalty = 20.0
        
        # Initialize attributes
        self.microgrid = None
        self.microgrid_generator = None
        self.load_data = None
        self.pv_data = None
        self.price_data = None
        
        # Initialize microgrid
        self._init_microgrid(microgrid_id)
        # If user specified an external CSV dataset, override time series
        try:
            if isinstance(self.dataset_csv_path, str) and len(self.dataset_csv_path) > 0:
                self._load_external_csv(self.dataset_csv_path, self.dataset_pv_join_wind)
                print(f"Loaded external CSV dataset: {self.dataset_csv_path}")
        except Exception as e:
            print(f"Warning: failed loading external CSV dataset: {e}")
        
        # Environment state and action spaces
        self._setup_spaces()
        
        # Episode tracking
        self.current_step = 0
        self.current_soc = 0.5  # Start at 50% SoC
        self.episode_data = None
        self.prev_action_kw = 0.0  # 新增：追蹤上一步動作
        
        # Statistics
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.soc_violations = 0
        self.action_violations = 0
        
    def _init_microgrid(self, microgrid_id: int):
        """初始化微電網（若無 python-microgrid，啟用合成模式並視為已初始化）"""
        if not PYTHON_MICROGRID_AVAILABLE:
            # 合成模式：標記為已初始化並準備合成資料
            self.microgrid = object()  # 非 None 的哨兵，表示已初始化
            self._generate_synthetic_data()
            print(f"Microgrid {microgrid_id} initialized (synthetic mode)")
            print(f"  - Battery capacity: {self.battery_capacity_kwh:.1f} kWh")
            print(f"  - Battery power: {self.battery_power_kw:.1f} kW")
            print(f"  - Episode length: {self.episode_length} steps")
            return
            
        try:
            # Try different ways to initialize microgrid
            if hasattr(MicrogridGenerator, 'generate'):
                # Old API
                self.microgrid_generator = MicrogridGenerator(nb_microgrid=1)
                self.microgrid_generator.generate(microgrid_id)
                self.microgrid = self.microgrid_generator.microgrids[microgrid_id]
            elif hasattr(MicrogridGenerator, 'create'):
                # New API
                self.microgrid_generator = MicrogridGenerator()
                self.microgrid = self.microgrid_generator.create(nb_microgrid=1)[microgrid_id]
            else:
                # Try direct creation
                self.microgrid = Microgrid()
                print("Warning: Using basic Microgrid instance")
            
            # Get time series data
            self._load_time_series_data()
            
            print(f"Microgrid {microgrid_id} initialized successfully")
            print(f"  - Battery capacity: {self.battery_capacity_kwh:.1f} kWh")
            print(f"  - Battery power: {self.battery_power_kw:.1f} kW")
            print(f"  - Episode length: {self.episode_length} steps")
            
        except Exception as e:
            print(f"Warning: Failed to initialize microgrid: {e}")
            print("Falling back to stub environment")
            self.microgrid = None
    
    def _load_time_series_data(self):
        """加載時間序列數據"""
        if self.microgrid is None:
            return
            
        try:
            # Get load and PV data
            self.load_data = self.microgrid.load_ts
            self.pv_data = self.microgrid.pv_ts
            
            # Get price data (if available)
            if hasattr(self.microgrid, 'price_ts'):
                self.price_data = self.microgrid.price_ts
            else:
                # Generate synthetic price data
                self.price_data = self._generate_synthetic_prices()
                
            print(f"Loaded time series data:")
            print(f"  - Load data: {len(self.load_data)} points")
            print(f"  - PV data: {len(self.pv_data)} points")
            print(f"  - Price data: {len(self.price_data)} points")
            
        except Exception as e:
            print(f"Warning: Failed to load time series data: {e}")
            self._generate_synthetic_data()

    def _load_external_csv(self, csv_path: str, pv_join_wind: bool = False):
        """
        Load external CSV dataset.
        - Standard format: index/Consumption/Solar/Wind
        - Raw acquisition format: timestamp, solar_p_mw/mppt_p_mw, with optional dataset_load_kw
        """
        import pandas as pd
        p = csv_path
        df = pd.read_csv(p)
        if 'Consumption' in df.columns:
            load = df['Consumption'].astype(float).values
            pv = None
            if 'Solar' in df.columns:
                pv = df['Solar'].astype(float).values
            if pv_join_wind and 'Wind' in df.columns:
                wind = df['Wind'].astype(float).fillna(0.0).values
                pv = (pv if pv is not None else 0.0) + wind
            if pv is None:
                pv = np.zeros_like(load)
            time_col = self.dataset_time_column or ('index' if 'index' in df.columns else None)
        else:
            # Raw acquisition format (e.g., DATA_Acquisition)
            pv_col = self.dataset_pv_column
            if not pv_col:
                if 'solar_p_mw' in df.columns:
                    pv_col = 'solar_p_mw'
                elif 'mppt_p_mw' in df.columns:
                    pv_col = 'mppt_p_mw'
            if not pv_col or pv_col not in df.columns:
                raise ValueError('CSV missing PV column (set dataset_pv_column)')
            pv_raw = df[pv_col].astype(float).fillna(0.0).values
            pv = pv_raw * float(self.dataset_power_scale)
            if self.dataset_load_kw is None:
                raise ValueError('CSV missing Consumption column and dataset_load_kw not provided')
            load = np.ones_like(pv) * float(self.dataset_load_kw)
            time_col = self.dataset_time_column or ('timestamp' if 'timestamp' in df.columns else None)

        # Build price time series with same length as load
        N = int(len(load))
        price = None
        try:
            if time_col and time_col in df.columns:
                t = pd.to_datetime(df[time_col], errors='coerce')
                # time-of-use: peak 8-18h, off-peak otherwise
                hour = t.dt.hour.fillna(0).astype(int).values
                base_price = 0.15
                daily_factor = np.where((hour >= 8) & (hour <= 18), 1.2, 0.8)
                price = np.clip(base_price * daily_factor, 0.05, None)
            else:
                # fallback: repeat 24h synthetic pattern
                daily = self._generate_synthetic_prices()  # 24 length
                reps = int(np.ceil(N / len(daily))) if len(daily) > 0 else 0
                price = (np.tile(daily, reps)[:N] if reps > 0 else np.ones(N) * 0.15)
        except Exception:
            price = np.ones(N, dtype=float) * 0.15
        # Assign series
        self.load_data = np.asarray(load, dtype=float)
        self.pv_data = np.asarray(pv, dtype=float)
        self.price_data = np.asarray(price, dtype=float)
    
    def _generate_synthetic_data(self):
        """生成合成數據 - 支持長期時間序列"""
        print(f"Generating synthetic microgrid data for {self.episode_length} steps...")

        if self.synthetic_hourly_hold and self.time_step < 1.0:
            steps_per_hour = max(1, int(round(1.0 / max(self.time_step, 1e-9))))
            hours = np.arange(24, dtype=float)
            pv_hourly = np.zeros_like(hours)
            start = int(self.synthetic_pv_start_hour)
            end = int(self.synthetic_pv_end_hour)
            if end > start:
                x = (hours - start) / max(1.0, (end - start))
                pv_hourly = self.synthetic_pv_peak_kw * np.sin(np.pi * np.clip(x, 0.0, 1.0))
                pv_hourly[(hours < start) | (hours > end)] = 0.0
            load_hourly = self.synthetic_load_base_kw + self.synthetic_load_amp_kw * (1.0 + np.sin(2 * np.pi * (hours - 7) / 24.0)) / 2.0
            price_hourly = np.ones_like(hours) * self.synthetic_price_base
            peak_mask = (hours >= self.synthetic_price_peak_start) & (hours <= self.synthetic_price_peak_end)
            price_hourly[peak_mask] = self.synthetic_price_peak

            pv_series = np.repeat(pv_hourly, steps_per_hour)
            load_series = np.repeat(load_hourly, steps_per_hour)
            price_series = np.repeat(price_hourly, steps_per_hour)

            total_steps = int(self.episode_length)
            if total_steps > len(pv_series):
                reps = int(np.ceil(total_steps / len(pv_series)))
                pv_series = np.tile(pv_series, reps)[:total_steps]
                load_series = np.tile(load_series, reps)[:total_steps]
                price_series = np.tile(price_series, reps)[:total_steps]
            else:
                pv_series = pv_series[:total_steps]
                load_series = load_series[:total_steps]
                price_series = price_series[:total_steps]

            self.load_data = np.maximum(load_series, 0.0)
            self.pv_data = np.maximum(pv_series, 0.0)
            self.price_data = np.maximum(price_series, 0.05)
            print("Synthetic hourly-hold data generated")
            return
        
        # Generate longer time series based on episode length
        if self.episode_length <= 24:
            # Daily pattern (24 hours)
            hours = np.arange(24)
            base_load = 30.0  # kW
            load_pattern = base_load + 20 * np.sin(2 * np.pi * (hours - 6) / 24) + 10 * np.random.randn(24)
            self.load_data = np.maximum(load_pattern, 5.0)  # Minimum 5 kW
            
            # Generate PV profile (solar pattern)
            solar_pattern = 40 * np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) + 5 * np.random.randn(24)
            self.pv_data = np.maximum(solar_pattern, 0.0)  # No negative PV
            
            # Generate price profile (time-of-use pricing)
            base_price = 0.15  # $/kWh
            price_pattern = base_price + 0.05 * np.sin(2 * np.pi * (hours - 12) / 24) + 0.02 * np.random.randn(24)
            self.price_data = np.maximum(price_pattern, 0.05)  # Minimum 5 cents/kWh
            
        elif self.episode_length <= 720:
            # Monthly pattern (30 days × 24 hours)
            self._generate_monthly_data()
        else:
            # Yearly pattern (365 days × 24 hours)
            self._generate_yearly_data()
        
        print("Synthetic data generated")
    
    def _generate_monthly_data(self):
        """生成月度數據 (30天 × 24小時 = 720步)"""
        days = np.arange(30)
        hours = np.arange(24)
        
        # Base load with weekly pattern
        base_load = 30.0  # kW
        weekly_pattern = base_load + 5 * np.sin(2 * np.pi * days / 7)  # Weekly variation
        
        # Daily load pattern
        daily_pattern = 20 * np.sin(2 * np.pi * (hours - 6) / 24)  # Daily variation
        
        # Combine patterns
        load_data = np.zeros(720)
        for day in range(30):
            for hour in range(24):
                idx = day * 24 + hour
                base = weekly_pattern[day]
                daily = daily_pattern[hour]
                noise = 8 * np.random.randn()
                load_data[idx] = base + daily + noise
        
        self.load_data = np.maximum(load_data, 5.0)  # Minimum 5 kW
        
        # PV data with seasonal variation
        pv_data = np.zeros(720)
        for day in range(30):
            for hour in range(24):
                idx = day * 24 + hour
                # Seasonal factor (assuming month 1-12)
                seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (day / 30 - 0.5))  # Peak in summer
                # Daily solar pattern
                solar_hour = hour - 6  # Solar noon at 12
                if 0 <= solar_hour <= 12:
                    solar_intensity = 40 * np.sin(np.pi * solar_hour / 12) * seasonal_factor
                else:
                    solar_intensity = 0
                
                pv_data[idx] = max(0, solar_intensity + 3 * np.random.randn())
        
        self.pv_data = pv_data
        
        # Price data with weekly and daily patterns
        price_data = np.zeros(720)
        base_price = 0.15  # $/kWh
        
        for day in range(30):
            for hour in range(24):
                idx = day * 24 + hour
                # Weekly pattern (weekend vs weekday)
                is_weekend = (day % 7) >= 5
                weekly_factor = 0.9 if is_weekend else 1.1
                
                # Daily pattern (peak hours)
                is_peak_hour = 8 <= hour <= 18
                daily_factor = 1.2 if is_peak_hour else 0.8
                
                # Base price with variations
                price = base_price * weekly_factor * daily_factor
                price += 0.02 * np.random.randn()  # Random noise
                price_data[idx] = max(0.05, price)
        
        self.price_data = price_data
    
    def _generate_yearly_data(self):
        """生成年度數據 (365天 × 24小時 = 8760步)"""
        days = np.arange(365)
        hours = np.arange(24)
        
        # Base load with seasonal and weekly patterns
        base_load = 30.0  # kW
        
        # Seasonal pattern (winter vs summer)
        seasonal_pattern = base_load + 15 * np.sin(2 * np.pi * (days - 172) / 365)  # Peak in summer
        
        # Weekly pattern
        weekly_pattern = 5 * np.sin(2 * np.pi * days / 7)  # Weekly variation
        
        # Daily load pattern
        daily_pattern = 20 * np.sin(2 * np.pi * (hours - 6) / 24)  # Daily variation
        
        # Combine all patterns
        load_data = np.zeros(8760)
        for day in range(365):
            for hour in range(24):
                idx = day * 24 + hour
                seasonal = seasonal_pattern[day]
                weekly = weekly_pattern[day]
                daily = daily_pattern[hour]
                noise = 10 * np.random.randn()
                load_data[idx] = seasonal + weekly + daily + noise
        
        self.load_data = np.maximum(load_data, 5.0)  # Minimum 5 kW
        
        # PV data with strong seasonal variation
        pv_data = np.zeros(8760)
        for day in range(365):
            for hour in range(24):
                idx = day * 24 + hour
                # Strong seasonal factor
                seasonal_factor = 0.3 + 0.7 * np.sin(2 * np.pi * (days[day] - 172) / 365)  # Peak in summer
                
                # Daily solar pattern
                solar_hour = hour - 6  # Solar noon at 12
                if 0 <= solar_hour <= 12:
                    solar_intensity = 50 * np.sin(np.pi * solar_hour / 12) * seasonal_factor
                else:
                    solar_intensity = 0
                
                pv_data[idx] = max(0, solar_intensity + 5 * np.random.randn())
        
        self.pv_data = pv_data
        
        # Price data with seasonal, weekly, and daily patterns
        price_data = np.zeros(8760)
        base_price = 0.15  # $/kWh
        
        for day in range(365):
            for hour in range(24):
                idx = day * 24 + hour
                
                # Seasonal pattern (higher prices in winter)
                seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (days[day] - 172) / 365)
                
                # Weekly pattern (weekend vs weekday)
                is_weekend = (day % 7) >= 5
                weekly_factor = 0.9 if is_weekend else 1.1
                
                # Daily pattern (peak hours)
                is_peak_hour = 8 <= hour <= 18
                daily_factor = 1.3 if is_peak_hour else 0.7
                
                # Base price with all variations
                price = base_price * seasonal_factor * weekly_factor * daily_factor
                price += 0.03 * np.random.randn()  # Random noise
                price_data[idx] = max(0.05, price)
        
        self.price_data = price_data
    
    def _generate_synthetic_prices(self):
        """生成合成電價數據"""
        hours = np.arange(24)
        base_price = 0.15  # $/kWh
        price_pattern = base_price + 0.05 * np.sin(2 * np.pi * (hours - 12) / 24) + 0.02 * np.random.randn(24)
        return np.maximum(price_pattern, 0.05)
    
    def _setup_spaces(self):
        """設置環境的狀態和動作空間"""
        # State space: [SoC, load, pv, price, hour, day_of_week]
        # SoC: 0-1, load: 0-100 kW, pv: 0-50 kW, price: 0-1 $/kWh, hour: 0-23, day: 0-6
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0, 0]),
            high=np.array([1.0, 100.0, 50.0, 1.0, 23, 6]),
            dtype=np.float32
        )
        
        # Action space: continuous battery power [-battery_power_kw, +battery_power_kw]
        self.action_space = spaces.Box(
            low=np.array([-self.battery_power_kw]),
            high=np.array([self.battery_power_kw]),
            dtype=np.float32
        )
    
    def _get_state(self) -> np.ndarray:
        """獲取當前狀態"""
        if self.microgrid is not None and self.episode_data is not None:
            # Use real data
            current_hour = self.current_step % 24
            current_day = (self.current_step // 24) % 7
            
            load = self.episode_data['load'][self.current_step] if self.current_step < len(self.episode_data['load']) else 0.0
            pv = self.episode_data['pv'][self.current_step] if self.current_step < len(self.episode_data['pv']) else 0.0
            price = self.episode_data['price'][self.current_step] if self.current_step < len(self.episode_data['price']) else 0.15
        else:
            # Use synthetic data
            current_hour = self.current_step % 24
            current_day = (self.current_step // 24) % 7
            
            load = self.load_data[current_hour] if hasattr(self, 'load_data') else 30.0
            pv = self.pv_data[current_hour] if hasattr(self, 'pv_data') else 20.0
            price = self.price_data[current_hour] if hasattr(self, 'price_data') else 0.15
        
        # Normalize price to 0-1 range
        price_normalized = np.clip(price / 0.5, 0.0, 1.0)  # Assuming max price is 0.5 $/kWh
        
        return np.array([
            self.current_soc,           # Battery SoC
            load,                       # Current load
            pv,                         # Current PV generation
            price_normalized,           # Normalized price
            current_hour,               # Hour of day
            current_day                 # Day of week
        ], dtype=np.float32)
    
    def _update_battery_soc(self, action: float) -> float:
        """更新電池SoC"""
        # Convert action from kW to kWh using effective dt
        dt_eff = float(getattr(self, '_effective_time_step', self.time_step))
        energy_change_kwh = action * dt_eff
        
        # Apply efficiency
        eta_eff = float(getattr(self, '_effective_efficiency', self.battery_efficiency))
        if energy_change_kwh > 0:  # Charging
            energy_change_kwh *= eta_eff
        else:  # Discharging
            eta_safe = eta_eff if eta_eff > 1e-9 else self.battery_efficiency
            energy_change_kwh /= eta_safe
        
        # Update SoC
        new_soc = self.current_soc + energy_change_kwh / self.battery_capacity_kwh
        
        # Check bounds
        low = float(getattr(self, 'soc_min_eff', self.soc_min))
        high = float(getattr(self, 'soc_max_eff', self.soc_max))
        if new_soc < low:
            self.soc_violations += 1
            new_soc = low
        elif new_soc > high:
            self.soc_violations += 1
            new_soc = high
        
        return new_soc

    def predict_soc_raw(self, soc: float, action: float) -> float:
        """以環境參數預估下一步 SoC（不夾限、不記違規）。
        Args:
            soc: 當前 SoC
            action: 充放電功率（kW，>0 充電，<0 放電）
        Returns:
            未夾限的下一步 SoC 預估值
        """
        energy_change_kwh = action * self.time_step
        if energy_change_kwh > 0:
            energy_change_kwh *= self.battery_efficiency
        else:
            energy_change_kwh /= self.battery_efficiency
        return soc + energy_change_kwh / self.battery_capacity_kwh
    
    def _calculate_reward(self, action: float, net_load: float, price: float) -> float:
        """計算獎勵函數 - 改進版本，鼓勵長期規劃"""
        reward = 0.0
        
        # 1. Energy arbitrage reward (主要獎勵)
        # 以 energy (kWh) 計算，確保與 $/kWh 一致
        dt = self.time_step
        if action < 0:  # Discharging (selling energy)
            reward += abs(action) * dt * price  # 收入（無偏置係數）
        else:  # Charging (buying energy)
            reward -= action * dt * price       # 成本（無偏置係數）

        # Throughput 懲罰（抑制無意義能量循環）
        reward -= 0.02 * abs(action) * dt
        
        # 2. Grid stability reward (電網穩定性)
        # 鼓勵負載平衡，懲罰極端情況
        net_load_penalty = -0.05 * (abs(net_load) / 100.0) ** 2  # 平方懲罰
        reward += net_load_penalty
        
        # 3. SoC management reward (SoC 管理 - 長期規劃)
        # 區間式：SoC 在 [0.3, 0.8] 給固定分數，區間外不加分（讓策略自由決策）
        soc_bonus_low, soc_bonus_high = 0.3, 0.8
        if soc_bonus_low <= self.current_soc <= soc_bonus_high:
            reward += 0.05
        
        # 4. SoC violation penalty (SoC 違反懲罰)
        if self.current_soc < self.soc_min or self.current_soc > self.soc_max:
            reward -= 2.0  # 適度懲罰，不要過重
        
        # 5. Action efficiency reward (動作效率獎勵 - 長期策略)
        # 鼓勵在低價時充電，高價時放電
        price_normalized = (price - 0.05) / 0.45  # 0.05 到 0.5 範圍
        if price_normalized > 0.5 and action < 0:  # 高價放電
            reward += 0.5
        elif price_normalized < 0.3 and action > 0:  # 低價充電
            reward += 0.3
        
        # 6. Action smoothness penalty (動作平滑性)
        # 懲罰過大的動作變化
        action_smoothness_penalty = -0.001 * (abs(action) / self.battery_power_kw) ** 2
        reward += action_smoothness_penalty
        
        # 7. Long-term planning reward (長期規劃獎勵)
        # 根據時間步長調整獎勵
        if self.episode_length > 24:
            # 對於長期episode，增加長期規劃的獎勵
            time_factor = min(1.0, self.current_step / (self.episode_length * 0.1))  # 前10%時間
            if time_factor < 1.0:
                # 在episode早期，鼓勵保守策略
                if abs(action) < self.battery_power_kw * 0.3:  # 保守動作
                    reward += 0.1 * (1.0 - time_factor)
                else:  # 激進動作
                    reward -= 0.05 * (1.0 - time_factor)
        
        # 8. Seasonal awareness reward (季節性意識獎勵)
        if self.episode_length > 720:  # 年度episode
            # 根據季節調整策略
            day_of_year = self.current_step // 24
            season = (day_of_year // 91) % 4  # 0:春, 1:夏, 2:秋, 3:冬
            
            if season == 1:  # 夏季 - 鼓勵儲存太陽能
                if action > 0 and self.current_soc < 0.7:  # 充電且SoC不高
                    reward += 0.2
            elif season == 3:  # 冬季 - 鼓勵節約用電
                if action < 0 and self.current_soc > 0.3:  # 放電且SoC不低
                    reward += 0.1
        
        return reward * self.reward_scaling
    
    def _calculate_reward_phase1(self, action: float, net_load: float, price: float) -> float:
        """Phase-1 獎勵函數 - 包含違規懲罰和 SafetyNet 介入懲罰"""
        reward = 0.0
        
        # 1. Energy arbitrage reward (主要獎勵)
        dt = self.time_step
        if action < 0:  # Discharging (selling energy)
            reward += abs(action) * dt * price
        else:  # Charging (buying energy)
            reward -= action * dt * price

        # Throughput 懲罰（抑制無意義能量循環）
        reward -= 0.02 * abs(action) * dt
        
        # 2. Grid stability reward (電網穩定性)
        net_load_penalty = -0.05 * (abs(net_load) / 100.0) ** 2
        reward += net_load_penalty
        
        # 3. SoC management reward (SoC 管理)
        soc_bonus_low, soc_bonus_high = 0.3, 0.8
        if soc_bonus_low <= self.current_soc <= soc_bonus_high:
            reward += 0.05
        
        # 4. SoC violation penalty：移至 step 中以未夾限 SoC 為準，避免雙重懲罰
        
        # 5. Action efficiency reward
        price_normalized = (price - 0.05) / 0.45
        if price_normalized > 0.5 and action < 0:  # 高價放電
            reward += 0.5
        elif price_normalized < 0.3 and action > 0:  # 低價充電
            reward += 0.3
        
        # 6. Action smoothness penalty
        action_smoothness_penalty = -0.001 * (abs(action) / self.battery_power_kw) ** 2
        reward += action_smoothness_penalty
        
        # 7. Long-term planning reward
        if self.episode_length > 24:
            time_factor = min(1.0, self.current_step / (self.episode_length * 0.1))
            if time_factor < 1.0:
                if abs(action) < self.battery_power_kw * 0.3:
                    reward += 0.1 * (1.0 - time_factor)
                else:
                    reward -= 0.05 * (1.0 - time_factor)
        
        # 8. Seasonal awareness reward
        if self.episode_length > 720:
            day_of_year = self.current_step // 24
            season = (day_of_year // 91) % 4
            
            if season == 1:  # 夏季
                if action > 0 and self.current_soc < 0.7:
                    reward += 0.2
            elif season == 3:  # 冬季
                if action < 0 and self.current_soc > 0.3:
                    reward += 0.1
        
        # Phase-1: 獎勵縮放
        return reward * self.reward_scaling
    
    def _calculate_reward_no_grid(self, action: float, net_load: float, load_kw: float, 
                                   pv_kw: float, price: float) -> float:
        """
        無電網交易版本的 reward function（孤島模式）
        
        核心目標：
        1. 最小化未供應負載（最重要）
        2. 最大化 PV 利用率
        3. SoC 管理（維持合適的 SoC）
        4. 能源效率（減少無意義循環）
        
        Args:
            action: 電池動作（kW），正數=充電，負數=放電
            net_load: 淨負載 = load_kw - pv_kw + action
            load_kw: 負載需求（kW）
            pv_kw: PV 發電（kW）
            price: 電價（$/kWh，可用於時段重要性加權）
        """
        reward = 0.0
        dt = self.time_step
        
        # ========== 1. 未供應負載懲罰（最重要，強烈懲罰）==========
        # 如果 net_load > 0，表示需要外部供電但無法獲得 → 未供應負載
        unserved_load = max(0.0, net_load)
        if unserved_load > 0:
            # 強烈懲罰：使用價格加權（高峰時段未供電懲罰更重）
            unserved_penalty = -10.0 * unserved_load * price * dt
            # 額外的平方懲罰，讓大額未供應負載懲罰更重
            unserved_penalty -= 5.0 * (unserved_load / max(load_kw, 1e-6)) ** 2 * dt
            reward += unserved_penalty
        
        # ========== 2. PV 利用率獎勵（鼓勵充分利用太陽能）==========
        if pv_kw > 1e-6:
            # PV 用於供給負載或充電的比例
            pv_used = min(pv_kw, load_kw + max(0, action))
            pv_utilization = pv_used / pv_kw
            pv_reward = 0.5 * pv_utilization
            reward += pv_reward
        
        # ========== 3. SoC 管理獎勵（維持合適的 SoC）==========
        soc_target_range = (0.4, 0.7)  # 理想 SoC 區間
        if soc_target_range[0] <= self.current_soc <= soc_target_range[1]:
            reward += 0.1  # 維持在理想區間
        elif self.current_soc < 0.2:
            # SoC 過低，可能無法應對未來高負載
            reward -= 0.05 * (0.2 - self.current_soc) / 0.2
        elif self.current_soc > 0.9:
            # SoC 過高，可能無法儲存多餘的 PV
            reward -= 0.02 * (self.current_soc - 0.9) / 0.1
        
        # ========== 4. SoC 違反懲罰（安全限制）==========
        # 注意：這裡只是預警，實際違反已在 step() 中處理並有額外懲罰
        if self.current_soc < self.soc_min or self.current_soc > self.soc_max:
            reward -= 5.0  # 強烈懲罰越界
        
        # ========== 5. 長期供電能力獎勵（基於時段預測）==========
        # 在 PV 發電高峰期（白天），鼓勵充電以應對夜間需求
        hour = self.current_step % 24
        if 8 <= hour <= 16:  # 白天時段（PV 發電期）
            if action > 0 and pv_kw > load_kw * 1.1:  # PV 多於負載時充電
                reward += 0.2
            # 避免白天過度放電（除非緊急需求）
            if action < -0.3 * self.battery_power_kw and pv_kw > load_kw:
                reward -= 0.1
        elif 18 <= hour <= 23 or 0 <= hour <= 6:  # 夜間/清晨時段（無 PV）
            # 負載多於 PV 時放電供應
            if action < 0 and load_kw > pv_kw * 1.1:
                reward += 0.15
            # 避免夜間過度充電（除非有大量多餘能源）
            if action > 0.3 * self.battery_power_kw and pv_kw < load_kw * 0.5:
                reward -= 0.05
        
        # ========== 6. 動作平滑性懲罰（減少快速變化）==========
        if hasattr(self, 'prev_action_kw'):
            action_change = abs(action - self.prev_action_kw)
            smoothness_penalty = -0.001 * (action_change / self.battery_power_kw) ** 2
            reward += smoothness_penalty
        
        # ========== 7. 能源效率懲罰（減少無意義循環）==========
        # 如果 SoC 已經很高，還繼續充電且沒有緊急需求 → 浪費
        if self.current_soc > 0.8 and action > 0.2 * self.battery_power_kw and load_kw < pv_kw * 0.5:
            reward -= 0.05
        # 如果 SoC 已經很低，還繼續放電且不是為了滿足負載 → 危險
        if self.current_soc < 0.2 and action < -0.2 * self.battery_power_kw and load_kw < pv_kw * 1.1:
            reward -= 0.1
        
        # ========== 8. Throughput 懲罰（抑制無意義能量循環）==========
        # 但權重較輕，因為在無電網模式下，充放電可能是必要的
        reward -= 0.01 * abs(action) * dt
        
        return reward * self.reward_scaling
    
    def _calculate_reward_no_grid_simplified(self, action: float, pv_kw: float, price: float) -> float:
        """
        無電網交易版本的 reward function（簡化版，不依賴負載資料）
        
        適用於：Data.txt 格式中沒有負載資料的情況
        
        核心目標：
        1. 最大化 PV 利用率
        2. SoC 管理（維持合適的 SoC）
        3. 能源效率（減少無意義循環）
        4. 時段策略（白天充電，夜間放電）
        
        Args:
            action: 電池動作（kW），正數=充電，負數=放電
            pv_kw: PV 發電（kW）
            price: 電價（$/kWh，可用於時段重要性加權）
        """
        reward = 0.0
        dt = self.time_step
        
        # ========== 1. PV 利用率獎勵（鼓勵充分利用太陽能）==========
        if pv_kw > 1e-6:
            # 如果 PV 大於電池功率，且正在充電 → 充分利用 PV
            if action > 0:
                # 鼓勵在 PV 發電時充電
                pv_utilization = min(1.0, abs(action) / max(pv_kw, 1e-6))
                reward += 0.5 * pv_utilization
                # 如果 PV 很大，且充電功率接近 PV → 更充分利用
                if pv_utilization > 0.8:
                    reward += 0.2
        
        # ========== 2. SoC 管理獎勵（維持合適的 SoC）==========
        soc_target_range = (0.4, 0.7)  # 理想 SoC 區間
        if soc_target_range[0] <= self.current_soc <= soc_target_range[1]:
            reward += 0.1  # 維持在理想區間
        elif self.current_soc < 0.2:
            # SoC 過低，可能無法應對未來高負載
            reward -= 0.05 * (0.2 - self.current_soc) / 0.2
        elif self.current_soc > 0.9:
            # SoC 過高，可能無法儲存多餘的 PV
            reward -= 0.02 * (self.current_soc - 0.9) / 0.1
        
        # ========== 3. SoC 違反懲罰（安全限制）==========
        if self.current_soc < self.soc_min or self.current_soc > self.soc_max:
            reward -= 5.0  # 強烈懲罰越界
        
        # ========== 4. 時段策略獎勵（基於時間）==========
        hour = self.current_step % 24
        if 8 <= hour <= 16:  # 白天時段（PV 發電期）
            if action > 0:  # 鼓勵充電
                reward += 0.2
                # 如果 PV 很大，充電更有利
                if pv_kw > 0.5 * self.battery_power_kw:
                    reward += 0.1
            # 避免白天過度放電（除非 SoC 很高）
            if action < -0.3 * self.battery_power_kw and self.current_soc < 0.7:
                reward -= 0.1
        elif 18 <= hour <= 23 or 0 <= hour <= 6:  # 夜間/清晨時段（無 PV）
            # 允許放電，但要保護低 SoC
            if action < 0 and self.current_soc > 0.3:
                reward += 0.1
            # 避免夜間過度放電（保護低 SoC）
            if action < -0.2 * self.battery_power_kw and self.current_soc < 0.3:
                reward -= 0.2
        
        # ========== 5. 動作平滑性懲罰（減少快速變化）==========
        if hasattr(self, 'prev_action_kw'):
            action_change = abs(action - self.prev_action_kw)
            smoothness_penalty = -0.001 * (action_change / self.battery_power_kw) ** 2
            reward += smoothness_penalty
        
        # ========== 6. 能源效率懲罰（減少無意義循環）==========
        # 如果 SoC 已經很高，還繼續充電且 PV 不大 → 浪費
        if self.current_soc > 0.8 and action > 0.2 * self.battery_power_kw and pv_kw < 0.3 * self.battery_power_kw:
            reward -= 0.05
        # 如果 SoC 已經很低，還繼續放電 → 危險
        if self.current_soc < 0.2 and action < -0.2 * self.battery_power_kw:
            reward -= 0.1
        
        # ========== 7. Throughput 懲罰（抑制無意義能量循環）==========
        reward -= 0.01 * abs(action) * dt
        
        return reward * self.reward_scaling
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置環境"""
        super().reset(seed=seed)
        
        # Reset episode state
        # 隨機起點：若有可用的 episode_data，從可切片區間內隨機抽起點
        # 先預設為 0，待資料準備好後再覆寫
        self.current_step = 0
        self.current_soc = 0.5  # Start at 50% SoC
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.soc_violations = 0
        self.action_violations = 0
        self.prev_action_kw = 0.0 # Reset previous action
        self._prev_exec_action_kw = 0.0
        # Reset effective bounds with drift if stress
        if self.stress_enable and self.stress_bounds_drift_std > 0.0:
            drift = float(np.random.randn()) * self.stress_bounds_drift_std
            # Shrink and shift within [soc_min, soc_max]
            base_low = self.soc_min + max(0.0, drift)
            base_high = self.soc_max - max(0.0, drift)
            margin = 1e-3
            self.soc_min_eff = float(np.clip(base_low, self.soc_min + margin, self.soc_max - 2*margin))
            self.soc_max_eff = float(np.clip(base_high, self.soc_min + 2*margin, self.soc_max - margin))
        else:
            self.soc_min_eff = self.soc_min
            self.soc_max_eff = self.soc_max
        # Initialize observation buffer for delayed/noisy SoC
        self._soc_obs_buffer = [self.current_soc] * (max(0, self.stress_soc_obs_delay) + 1)
        
        # Prepare episode data（若尚未載入，先產生資料）
        if self.microgrid is not None and self.use_real_data and self.load_data is not None:
            # Use real microgrid data
            self.episode_data = {
                'load': self.load_data[:self.episode_length],
                'pv': self.pv_data[:self.episode_length],
                'price': self.price_data[:self.episode_length]
            }
        else:
            # Use synthetic data
            if self.load_data is None:
                self._generate_synthetic_data()
            
            self.episode_data = {
                'load': self.load_data[:self.episode_length] if self.load_data is not None else np.ones(self.episode_length) * 30.0,
                'pv': self.pv_data[:self.episode_length] if self.pv_data is not None else np.ones(self.episode_length) * 20.0,
                'price': self.price_data[:self.episode_length] if self.price_data is not None else np.ones(self.episode_length) * 0.15
            }
        
        # 隨機/固定起點抽樣：在資料已備妥時，從 [0, len-episode_length] 抽樣起點
        try:
            total_len = int(min(len(self.episode_data['load']), len(self.episode_data['pv']), len(self.episode_data['price'])))
            if total_len >= int(self.episode_length) and int(self.episode_length) > 0:
                max_start_global = max(0, total_len - int(self.episode_length))
                if self.fixed_start_idx is not None:
                    start_idx = int(max(0, min(self.fixed_start_idx, max_start_global)))
                else:
                    if self.train_window_hours is not None:
                        # 限制隨機起點僅落在前 train_window_hours 範圍內
                        max_start = int(max(0, min(self.train_window_hours, max_start_global)))
                    else:
                        max_start = max_start_global
                    start_idx = int(np.random.randint(0, max_start + 1))
                # 重新切片本回合視窗
                self.episode_data = {
                    'load': self.episode_data['load'][start_idx:start_idx + self.episode_length],
                    'pv': self.episode_data['pv'][start_idx:start_idx + self.episode_length],
                    'price': self.episode_data['price'][start_idx:start_idx + self.episode_length],
                }
                self.current_step = 0
        except Exception:
            # 若失敗，維持從 0 開始
            self.current_step = 0

        # Weather variation（僅對合成資料生效）
        if not self.use_real_data and self.episode_data is not None:
            pv = self.episode_data.get('pv', None)
            load = self.episode_data.get('load', None)
            if pv is not None and self.weather_pv_scale_std > 0.0:
                pv_scale = max(0.0, 1.0 + np.random.randn() * self.weather_pv_scale_std)
                pv = pv * pv_scale
            if load is not None and self.weather_load_scale_std > 0.0:
                load_scale = max(0.0, 1.0 + np.random.randn() * self.weather_load_scale_std)
                load = load * load_scale
            if pv is not None and self.weather_pv_noise_std > 0.0:
                pv = pv * (1.0 + np.random.randn(len(pv)) * self.weather_pv_noise_std)
            if pv is not None:
                self.episode_data['pv'] = np.maximum(pv, 0.0)
            if load is not None:
                self.episode_data['load'] = np.maximum(load, 0.0)

        # Get initial state
        initial_state = self._get_state()
        
        info = {
            'soc': self.current_soc,
            'step': self.current_step,
            'load': self.episode_data['load'][0],
            'pv': self.episode_data['pv'][0],
            'price': self.episode_data['price'][0]
        }
        
        return initial_state, info
    
    def step(self, action: List[float]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """執行一步環境互動"""
        if self.current_step >= self.episode_length:
            return self._get_state(), 0.0, True, False, {}
        
        # 解析動作（kW）
        action_kw = float(action[0])
        
        # Phase-1: 檢查 ramp limit
        if self.ramp_limit_kw is not None:
            action_diff = abs(action_kw - self.prev_action_kw)
            if action_diff > self.ramp_limit_kw:
                # 違反爬坡限制，記錄並限制動作
                self.action_violations += 1
                if action_kw > self.prev_action_kw:
                    action_kw = self.prev_action_kw + self.ramp_limit_kw
                else:
                    action_kw = self.prev_action_kw - self.ramp_limit_kw
        
        # 應用 stress：動作滯後與外部 Pmax 收緊
        applied_action_kw = action_kw
        if self.stress_enable and self.stress_action_lag_alpha > 0.0:
            alpha = float(np.clip(self.stress_action_lag_alpha, 0.0, 0.99))
            applied_action_kw = alpha * self._prev_exec_action_kw + (1.0 - alpha) * action_kw
        # 外部 Pmax 收斂
        if self.stress_enable and self.stress_external_pmax_shrink_prob > 0.0:
            if np.random.rand() < self.stress_external_pmax_shrink_prob:
                shrink = float(np.clip(self.stress_external_pmax_shrink_factor, 0.1, 1.0))
                applied_action_kw = float(np.clip(applied_action_kw, -self.battery_power_kw * shrink, self.battery_power_kw * shrink))
        
        # 功率損耗：充電時需要更多功率，放電時輸出功率減少
        if self.stress_enable and self.stress_power_loss_ratio > 0.0:
            loss_ratio = float(self.stress_power_loss_ratio)
            if applied_action_kw > 0:  # 充電
                # 實際需要的功率 = 命令功率 / (1 - loss_ratio)
                applied_action_kw = applied_action_kw / max(1e-6, 1.0 - loss_ratio)
            elif applied_action_kw < 0:  # 放電
                # 實際輸出的功率 = 命令功率 * (1 - loss_ratio)
                applied_action_kw = applied_action_kw * (1.0 - loss_ratio)
            # 限制在物理功率範圍內
            applied_action_kw = float(np.clip(applied_action_kw, -self.battery_power_kw, self.battery_power_kw))
        
        # 設定有效 dt 與效率（提供給 _update_battery_soc 使用）
        if self.stress_enable:
            dt_jitter = 1.0 + float(np.random.randn()) * self.stress_dt_jitter_std
            self._effective_time_step = max(1e-3, float(self.time_step) * dt_jitter)
            eta_noise = 1.0 + float(np.random.randn()) * self.stress_efficiency_noise_std
            self._effective_efficiency = float(np.clip(self.battery_efficiency * eta_noise, 1e-3, 1.0))
        else:
            self._effective_time_step = float(self.time_step)
            self._effective_efficiency = float(self.battery_efficiency)

        # 更新電池 SoC（使用 applied_action_kw）
        old_soc = self.current_soc
        # 預估未夾限的下一步 SoC，用於計算實際違規能量（kWh）
        try:
            soc_next_raw = float(self.predict_soc_raw(old_soc, applied_action_kw))
        except Exception:
            soc_next_raw = old_soc
        # 計算違規能量（超出邊界的 kWh），稍後納入懲罰
        energy_violate_kwh = 0.0
        if soc_next_raw < self.soc_min:
            energy_violate_kwh = (self.soc_min - soc_next_raw) * self.battery_capacity_kwh
        elif soc_next_raw > self.soc_max:
            energy_violate_kwh = (soc_next_raw - self.soc_max) * self.battery_capacity_kwh

        # 物理護欄（僅在 hard_guard 開啟時啟用）：若預測越界，反解動作貼到邊界，避免實際越界
        if self.hard_guard and (soc_next_raw < self.soc_min or soc_next_raw > self.soc_max):
            dt = float(self.time_step)
            eta = float(self.battery_efficiency)
            cap = float(self.battery_capacity_kwh)
            target_soc = self.soc_min if soc_next_raw < self.soc_min else self.soc_max
            delta_e = (target_soc - old_soc) * cap  # kWh
            if delta_e >= 0:
                action_kw = delta_e / (dt * eta)
            else:
                action_kw = delta_e * eta / dt
            action_kw = float(np.clip(action_kw, -self.battery_power_kw, self.battery_power_kw))
            soc_next_raw = float(self.predict_soc_raw(old_soc, action_kw))

        self.current_soc = self._update_battery_soc(applied_action_kw)
        self._prev_exec_action_kw = applied_action_kw
        
        # SoC 違規累計已在 _update_battery_soc 中處理，這裡不再重複加計
        
        # 計算淨負載和電價
        load_kw = self.episode_data['load'][self.current_step]
        pv_kw = self.episode_data['pv'][self.current_step]
        price = self.episode_data['price'][self.current_step]
        
        net_load = load_kw - pv_kw + applied_action_kw
        
        # 計算財務指標
        if applied_action_kw < 0:  # 放電（賣電）
            revenue = abs(applied_action_kw) * self.time_step * price
            cost = 0.0
        else:  # 充電（買電）
            revenue = 0.0
            cost = applied_action_kw * self.time_step * price
        
        self.total_revenue += revenue
        self.total_cost += cost
        
        # 計算獎勵（根據是否允許電網交易選擇不同版本）
        if self.allow_grid_trading:
            reward = self._calculate_reward_phase1(action_kw, net_load, price)
        else:
            # 無電網交易模式（孤島模式）
            # 由於負載資料是假的（電子負載），直接使用簡化版本
            # 簡化版本專注於 PV 利用率和 SoC 管理，不依賴負載資料
            reward = self._calculate_reward_no_grid_simplified(action_kw, pv_kw, price)
        # 追加：對未夾限 SoC 的實際越界能量施以強懲罰（不受縮放影響）
        if energy_violate_kwh > 0.0:
            penalty_per_kwh = float(getattr(self, 'realized_violation_penalty', 20.0))
            scale_guard = max(float(self.reward_scaling), 1e-9)
            reward -= (penalty_per_kwh * energy_violate_kwh) / scale_guard
        
        # 更新狀態
        self.current_step += 1
        self.prev_action_kw = action_kw  # 記錄當前動作
        
        # 檢查是否結束
        done = self.current_step >= self.episode_length
        
        # 返回資訊
        info = {
            'total_revenue': self.total_revenue,
            'total_cost': self.total_cost,
            'soc_violations': self.soc_violations,
            'action_violations': self.action_violations,
            'current_soc': self.current_soc,
            'net_load': net_load,
            'price': price,
            'step': self.current_step,
            'load': load_kw,
            'pv': pv_kw,
            'action_kw': action_kw,
            'applied_action_kw': applied_action_kw
        }
        
        # 更新觀測 SoC 緩衝
        if self.stress_enable and self.stress_soc_obs_delay > 0:
            self._soc_obs_buffer.pop(0)
            self._soc_obs_buffer.append(self.current_soc)

        return self._get_state(), reward, done, False, info
    
    def render(self):
        """渲染環境（可選）"""
        pass
    
    def close(self):
        """關閉環境"""
        pass


class MicrogridEnvWrapper(gym.Env):
    """
    微電網環境的包裝器，提供統一的接口
    """
    
    def __init__(self, env: MicrogridEnvironment):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    @property
    def unwrapped(self):
        return self.env


def create_microgrid_env(
    microgrid_id: int = 0,
    episode_length: int = 24,
    time_step: float = 1.0,
    battery_capacity_kwh: float = 100.0,
    battery_power_kw: float = 50.0,
    use_real_data: bool = True,
    ramp_limit_kw: float = None,  # 新增參數
    hard_guard: bool = False,
    allow_grid_trading: bool = True,  # 是否允許電網交易
    **kwargs
) -> MicrogridEnvironment:
    """創建微電網環境"""
    return MicrogridEnvironment(
        microgrid_id=microgrid_id,
        episode_length=episode_length,
        time_step=time_step,
        allow_grid_trading=allow_grid_trading,
        battery_capacity_kwh=battery_capacity_kwh,
        battery_power_kw=battery_power_kw,
        use_real_data=use_real_data,
        ramp_limit_kw=ramp_limit_kw,  # 新增參數
        hard_guard=hard_guard,
        **kwargs
    ) 