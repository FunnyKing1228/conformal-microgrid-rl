"""
運行時配置管理
允許在程式運行時動態更新參數（透過配置文件）
"""

import json
import os
import time
from typing import Optional, Dict, Any


class RuntimeConfig:
    """運行時配置管理器"""
    
    def __init__(self, config_file: str = "runtime_config.json"):
        """
        初始化運行時配置管理器
        
        參數:
            config_file: 配置文件路徑
        """
        self.config_file = config_file
        self.last_modified = 0.0
        self.cached_config: Optional[Dict[str, Any]] = None
        
    def get_config(self) -> Dict[str, Any]:
        """
        讀取配置文件（會檢查文件修改時間，如果更新則重新讀取）
        
        返回:
            配置字典
        """
        if not os.path.exists(self.config_file):
            # 如果配置文件不存在，返回空配置
            return {}
        
        try:
            # 檢查文件修改時間
            mtime = os.path.getmtime(self.config_file)
            if mtime > self.last_modified or self.cached_config is None:
                # 文件已更新或尚未緩存，重新讀取
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.cached_config = json.load(f)
                self.last_modified = mtime
            return self.cached_config.copy() if self.cached_config else {}
        except (json.JSONDecodeError, IOError) as e:
            print(f"[運行時配置] 讀取配置文件失敗: {e}")
            return self.cached_config.copy() if self.cached_config else {}
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        更新配置文件
        
        參數:
            updates: 要更新的配置項
        
        返回:
            是否成功
        """
        try:
            # 讀取現有配置
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # 更新配置
            config.update(updates)
            
            # 寫回文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # 清除緩存，強制下次讀取時重新載入
            self.last_modified = 0.0
            self.cached_config = None
            
            return True
        except (IOError, json.JSONEncodeError) as e:
            print(f"[運行時配置] 更新配置文件失敗: {e}")
            return False


# 全局實例（可選）
_global_runtime_config: Optional[RuntimeConfig] = None

def get_runtime_config(config_file: str = "runtime_config.json") -> RuntimeConfig:
    """獲取全局運行時配置實例"""
    global _global_runtime_config
    if _global_runtime_config is None:
        _global_runtime_config = RuntimeConfig(config_file)
    return _global_runtime_config



