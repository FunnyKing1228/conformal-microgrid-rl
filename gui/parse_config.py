"""
解析配置管理模組

提供可配置的解析規則，允許通過 config_parse.json 調整解析邏輯，無需重新打包。
"""

import os
import json
from typing import Dict, Any, Optional

# 預設配置（如果配置文件不存在或損壞時使用）
DEFAULT_CONFIG = {
    "data_file": {
        "mppt_parsing": {
            "units": {
                "solar_v": {"multiplier": 0.01},
                "solar_i": {"multiplier": 1.0},  # 直接記錄原始數值（毫安），不除以1000
                "solar_p": {"multiplier": 1.0},  # 直接記錄原始數值（毫瓦），不除以1000
                "mppt_v": {"multiplier": 0.01},
                "mppt_i": {"multiplier": 1.0},   # 直接記錄原始數值（毫安），不除以1000
                "mppt_p": {"multiplier": 1.0},   # 直接記錄原始數值（毫瓦），不除以1000
            }
        },
        "battery_parsing": {
            "units": {
                "soc": {"multiplier": 0.1},
                "bv": {"multiplier": 0.01},
                "bi": {"multiplier": 1.0},  # 直接記錄原始數值（毫安），不除以1000
                "temp": {"multiplier": 0.1},
                "speed": {"multiplier": 0.1},
            }
        }
    },
    "command_file": {
        "power_units": {
            "conversion": {"multiplier": 1000.0}
        },
        "flow_units": {
            "conversion": {"multiplier": 1.0}
        }
    },
    "logging": {
        "enable_parse_logging": False,
        "log_file": "parse_debug.log",
        "log_level": "INFO",
        "log_parsed_values": False,
        "log_raw_values": False,
    }
}


class ParseConfig:
    """解析配置管理器"""
    
    _instance: Optional['ParseConfig'] = None
    _config: Dict[str, Any] = {}
    _config_path: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self._load_config()
    
    def _find_config_file(self) -> Optional[str]:
        """尋找配置文件（支援 EXE 模式和開發模式）"""
        import sys
        
        if getattr(sys, 'frozen', False):
            # EXE 模式：在 EXE 所在目錄尋找
            exe_dir = os.path.dirname(sys.executable)
            config_path = os.path.join(exe_dir, "config_parse.json")
        else:
            # 開發模式：在專案根目錄尋找
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            config_path = os.path.join(project_root, "config_parse.json")
        
        return config_path if os.path.exists(config_path) else None
    
    def _load_config(self):
        """載入配置文件"""
        config_path = self._find_config_file()
        self._config_path = config_path
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                # 驗證配置完整性（合併預設值）
                self._config = self._merge_defaults(self._config, DEFAULT_CONFIG)
            except Exception as e:
                # 配置文件損壞，使用預設值
                self._config = DEFAULT_CONFIG.copy()
        else:
            # 配置文件不存在，使用預設值
            self._config = DEFAULT_CONFIG.copy()
    
    def _merge_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """合併配置與預設值（深度合併）"""
        result = defaults.copy()
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_defaults(value, result[key])
            else:
                result[key] = value
        return result
    
    def reload_config(self):
        """重新載入配置文件（用於運行時更新）"""
        self._config = {}
        self._load_config()
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        取得配置值（支援點號分隔的路徑，例如 'data_file.mppt_parsing.units.solar_v.multiplier'）
        """
        keys = path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_mppt_multiplier(self, field: str) -> float:
        """取得 MPPT 欄位的單位轉換倍數"""
        path = f"data_file.mppt_parsing.units.{field}.multiplier"
        return self.get(path, 1.0)
    
    def get_battery_multiplier(self, field: str) -> float:
        """取得電池欄位的單位轉換倍數"""
        path = f"data_file.battery_parsing.units.{field}.multiplier"
        return self.get(path, 1.0)
    
    def get_power_conversion_multiplier(self) -> float:
        """取得功率單位轉換倍數（W -> mW）"""
        return self.get("command_file.power_units.conversion.multiplier", 1000.0)
    
    def get_flow_conversion_multiplier(self) -> float:
        """取得流速單位轉換倍數（通常為 1.0）"""
        return self.get("command_file.flow_units.conversion.multiplier", 1.0)
    
    def is_logging_enabled(self) -> bool:
        """檢查是否啟用解析日誌"""
        return self.get("logging.enable_parse_logging", False)
    
    def get_log_file(self) -> str:
        """取得日誌檔案路徑"""
        return self.get("logging.log_file", "parse_debug.log")
    
    def should_log_parsed_values(self) -> bool:
        """是否記錄解析後的值"""
        return self.get("logging.log_parsed_values", False)
    
    def should_log_raw_values(self) -> bool:
        """是否記錄原始值"""
        return self.get("logging.log_raw_values", False)
    
    def get_config_path(self) -> Optional[str]:
        """取得配置文件路徑（用於顯示給用戶）"""
        return self._config_path


# 全域配置實例
_config_manager = ParseConfig()


def get_config() -> ParseConfig:
    """取得配置管理器實例"""
    return _config_manager



