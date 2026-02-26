#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
負載計算器模組：計算最少需要開啟幾顆負載，讓系統（電池+太陽能）剛好可以供給

功能：
- 根據太陽能發電量、電池可用功率、負載規格，計算最少需要開啟的負載顆數
- 目標：讓系統剛好可以供給負載（避免過載或浪費）
"""

from typing import Optional, Tuple


class LoadCalculator:
    """負載計算器"""
    
    def __init__(self, load_power_per_unit_w: float):
        """
        初始化負載計算器
        
        Args:
            load_power_per_unit_w: 每顆負載的功率（W），例如 1000W = 1kW
        """
        if load_power_per_unit_w <= 0:
            raise ValueError(f"負載功率必須大於 0，當前值: {load_power_per_unit_w}W")
        self.load_power_per_unit_w = load_power_per_unit_w
    
    def calculate_load_count(self, 
                            pv_power_w: float,
                            battery_available_power_w: float,
                            max_load_count: int = 10) -> int:
        """
        計算最少需要開啟的負載顆數
        
        邏輯：
        1. 計算系統可用功率 = PV 功率 + 電池可用功率
        2. 計算最少負載顆數 = ceil(系統可用功率 / 每顆負載功率)
        3. 限制在 0 到 max_load_count 之間
        
        Args:
            pv_power_w: 太陽能發電功率（W），必須 >= 0
            battery_available_power_w: 電池可用功率（W），正值表示可放電，負值表示需要充電
                                       例如：電池可以放電 5000W，則傳入 5000
                                       例如：電池需要充電 2000W，則傳入 -2000
            max_load_count: 最大負載顆數（預設 10）
        
        Returns:
            需要開啟的負載顆數（0 ~ max_load_count）
        """
        # 參數驗證
        if pv_power_w < 0:
            raise ValueError(f"PV 功率不能為負數，當前值: {pv_power_w}W")
        if max_load_count < 0:
            raise ValueError(f"最大負載顆數不能為負數，當前值: {max_load_count}")
        
        # 計算系統可用功率（供給負載的功率）
        # 系統可用功率 = PV 功率 + 電池可用功率
        # 如果電池可用功率為負（需要充電），則從 PV 功率中扣除
        system_available_power_w = pv_power_w + battery_available_power_w
        
        # 如果系統可用功率為負或零，則不需要開啟負載
        if system_available_power_w <= 0:
            return 0
        
        # 計算需要的負載顆數（向上取整，確保可以消耗所有可用功率）
        import math
        load_count = math.ceil(system_available_power_w / self.load_power_per_unit_w)
        
        # 限制在 0 到 max_load_count 之間
        load_count = max(0, min(load_count, max_load_count))
        
        return load_count
    
    def calculate_load_count_from_soc(self,
                                     pv_power_w: float,
                                     battery_soc_percent: float,
                                     battery_capacity_kwh: float,
                                     battery_max_power_kw: float,
                                     soc_min: float = 0.1,
                                     soc_max: float = 0.9,
                                     battery_efficiency: float = 0.95) -> int:
        """
        根據電池 SoC 計算負載顆數（更智能的方法）
        
        邏輯：
        1. 根據 SoC 判斷電池可用功率
           - SoC > soc_max (0.9): 電池接近滿電，可以大量放電
           - SoC < soc_min (0.1): 電池接近空電，不能放電，可能需要充電
           - SoC 在範圍內: 根據 SoC 比例計算可用功率
        
        2. 計算負載顆數
        
        Args:
            pv_power_w: 太陽能發電功率（W）
            battery_soc_percent: 電池 SoC（%），0-100
            battery_capacity_kwh: 電池容量（kWh）
            battery_max_power_kw: 電池最大功率（kW）
            soc_min: SoC 下限（0-1），預設 0.1 (10%)
            soc_max: SoC 上限（0-1），預設 0.9 (90%)
            battery_efficiency: 電池效率（0-1），預設 0.95 (95%)
        
        Returns:
            需要開啟的負載顆數
        """
        # 將 SoC 轉換為 0-1 範圍
        soc_frac = battery_soc_percent / 100.0
        
        # 計算電池可用功率（W）
        battery_available_power_w = 0.0
        
        if soc_frac > soc_max:
            # 電池接近滿電，可以大量放電
            # 可用功率 = 最大功率（考慮效率）
            battery_available_power_w = battery_max_power_kw * 1000.0 * battery_efficiency
        elif soc_frac < soc_min:
            # 電池接近空電，不能放電，可能需要充電
            # 可用功率 = 0（不能放電），如果 PV 有剩餘，可以用來充電
            battery_available_power_w = 0.0
        else:
            # SoC 在範圍內，根據 SoC 比例計算可用功率
            # 可用功率 = 最大功率 × (SoC - soc_min) / (soc_max - soc_min) × 效率
            soc_range = soc_max - soc_min
            soc_in_range = (soc_frac - soc_min) / soc_range if soc_range > 0 else 0.5
            battery_available_power_w = battery_max_power_kw * 1000.0 * soc_in_range * battery_efficiency
        
        # 調用主要計算函數
        return self.calculate_load_count(pv_power_w, battery_available_power_w)
    
    def get_load_power(self, load_count: int) -> float:
        """
        根據負載顆數計算總負載功率
        
        Args:
            load_count: 負載顆數（0 ~ max_load_count）
        
        Returns:
            總負載功率（W）
        """
        return load_count * self.load_power_per_unit_w


def calculate_min_load_count_simple(pv_power_w: float,
                                    battery_available_power_w: float,
                                    load_power_per_unit_w: float,
                                    max_load_count: int = 10) -> int:
    """
    簡單的負載計算函數（獨立函數，方便直接調用）
    
    Args:
        pv_power_w: 太陽能發電功率（W）
        battery_available_power_w: 電池可用功率（W）
        load_power_per_unit_w: 每顆負載的功率（W）
        max_load_count: 最大負載顆數
    
    Returns:
        需要開啟的負載顆數（0 ~ max_load_count）
    """
    calculator = LoadCalculator(load_power_per_unit_w)
    return calculator.calculate_load_count(pv_power_w, battery_available_power_w, max_load_count)


if __name__ == "__main__":
    # 測試負載計算器
    print("負載計算器測試")
    print("=" * 60)
    
    # 測試案例 1：基本計算
    calculator = LoadCalculator(load_power_per_unit_w=1000.0)  # 每顆負載 1kW
    
    # 案例 1：PV 5kW，電池可放電 3kW，每顆負載 1kW
    # 系統可用功率 = 5 + 3 = 8kW，需要 8 顆負載
    load_count = calculator.calculate_load_count(
        pv_power_w=5000.0,
        battery_available_power_w=3000.0,
        max_load_count=10
    )
    print(f"案例 1: PV=5kW, 電池可用=3kW, 每顆負載=1kW")
    print(f"  計算結果: 需要 {load_count} 顆負載")
    print(f"  總負載功率: {calculator.get_load_power(load_count):.1f}W ({calculator.get_load_power(load_count)/1000:.1f}kW)")
    print()
    
    # 案例 2：PV 2kW，電池需要充電 1kW（可用功率為負）
    # 系統可用功率 = 2 - 1 = 1kW，需要 1 顆負載
    load_count = calculator.calculate_load_count(
        pv_power_w=2000.0,
        battery_available_power_w=-1000.0,  # 負值表示需要充電
        max_load_count=10
    )
    print(f"案例 2: PV=2kW, 電池需要充電=1kW, 每顆負載=1kW")
    print(f"  計算結果: 需要 {load_count} 顆負載")
    print(f"  總負載功率: {calculator.get_load_power(load_count):.1f}W ({calculator.get_load_power(load_count)/1000:.1f}kW)")
    print()
    
    # 案例 3：PV 0.5kW，電池可用 0.3kW，每顆負載 1kW
    # 系統可用功率 = 0.5 + 0.3 = 0.8kW，需要 1 顆負載（向上取整）
    load_count = calculator.calculate_load_count(
        pv_power_w=500.0,
        battery_available_power_w=300.0,
        max_load_count=10
    )
    print(f"案例 3: PV=0.5kW, 電池可用=0.3kW, 每顆負載=1kW")
    print(f"  計算結果: 需要 {load_count} 顆負載")
    print(f"  總負載功率: {calculator.get_load_power(load_count):.1f}W ({calculator.get_load_power(load_count)/1000:.1f}kW)")
    print()
    
    # 測試案例 4：根據 SoC 計算
    load_count = calculator.calculate_load_count_from_soc(
        pv_power_w=3000.0,
        battery_soc_percent=80.0,  # 80% SoC
        battery_capacity_kwh=10.0,
        battery_max_power_kw=5.0,
        soc_min=0.1,
        soc_max=0.9,
        battery_efficiency=0.95
    )
    print(f"案例 4 (根據 SoC): PV=3kW, SoC=80%, 電池容量=10kWh, 最大功率=5kW")
    print(f"  計算結果: 需要 {load_count} 顆負載")
    print(f"  總負載功率: {calculator.get_load_power(load_count):.1f}W ({calculator.get_load_power(load_count)/1000:.1f}kW)")
    print()



