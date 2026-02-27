"""
SOH Predictor — 電池健康狀態預測模組
====================================
整合自 SOH_Predictor 專案（Transformer-based SoH prediction）

用途：
  在 RL 環境中，根據充電段的 (time, voltage, current) 資料
  即時推論電池的 SoH (0~1)。

使用方式：
  from core.soh_predictor import SoHPredictor

  predictor = SoHPredictor()  # 自動載入預設模型
  soh = predictor.predict_from_csv("path/to/cycle.csv")
  soh = predictor.predict_from_arrays(time_s, voltage_v, current_a)
"""

from .inference import SoHPredictor

__all__ = ["SoHPredictor"]
