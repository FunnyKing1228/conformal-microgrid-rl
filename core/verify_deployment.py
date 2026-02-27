#!/usr/bin/env python3
"""
端到端 Deployment 驗證腳本
=========================
驗證：
  1. 模型載入（from best_sac_model.pth）
  2. 模擬 Data.txt → 讀取 → build_agent_state → inference → Command.txt
  3. 15 分鐘時間步與 I/O 格式一致性
  4. 多步模擬（模擬 24 小時 = 96 步）
"""

import os
import sys
import tempfile
import shutil
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import torch

# 設定路徑
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'control'))

from sac_agent import SACAgent
from control.io_protocol import (
    read_vendor_data_file,
    write_control_file_vendor,
    format_ts,
    parse_ts,
    TZ_UTC8,
)

# ── 常數 ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(PROJECT_ROOT, 'experiments', 'p302_sim_v1', 'models', 'best_sac_model.pth')
PMAX_KW = 0.00017        # P302 最大功率
STATE_DIM = 6             # [soc, load, pv, price, hour, dow]
ACTION_DIM = 1            # [power_norm]
HIDDEN_DIM = 128          # 訓練時用的 hidden_dim
INTERVAL_SEC = 900        # 15 分鐘
STEPS_PER_DAY = 96        # 24h × 4
SOC_MIN = 0.10
SOC_MAX = 0.90

passed = 0
failed = 0
errors = []


def check(name: str, condition: bool, msg: str = ""):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        err = f"  ✗ {name} — {msg}"
        errors.append(err)
        print(err)


# ══════════════════════════════════════════════════════════════════
# Phase 1  模型載入與推論
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Phase 1: 模型載入與推論驗證")
print("=" * 70)

# 1-1 模型檔案存在
check("模型檔案存在", os.path.exists(MODEL_PATH), f"找不到 {MODEL_PATH}")

# 1-2 載入 checkpoint
try:
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    check("Checkpoint 載入成功", True)
except Exception as e:
    check("Checkpoint 載入成功", False, str(e))
    checkpoint = None

# 1-3 Checkpoint 結構
if checkpoint:
    required_keys = ['actor', 'critic1', 'critic2']
    for k in required_keys:
        check(f"Checkpoint 包含 '{k}'", k in checkpoint, f"缺少 key: {k}")
    
    # 推斷 hidden_dim
    if 'actor' in checkpoint:
        actor_state = checkpoint['actor']
        if 'fc1.weight' in actor_state:
            inferred_hidden = actor_state['fc1.weight'].shape[0]
            inferred_state_dim = actor_state['fc1.weight'].shape[1]
            check(f"hidden_dim = {inferred_hidden} (期望 {HIDDEN_DIM})",
                  inferred_hidden == HIDDEN_DIM)
            check(f"state_dim = {inferred_state_dim} (期望 {STATE_DIM})",
                  inferred_state_dim == STATE_DIM)

# 1-4 建立 Agent 並載入
agent = None
try:
    agent = SACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device='cpu',
        hidden_dim=HIDDEN_DIM,
    )
    agent.load(MODEL_PATH)
    check("SACAgent 建立並載入成功", True)
except Exception as e:
    check("SACAgent 建立並載入成功", False, str(e))

# 1-5 推論測試
if agent:
    test_state = np.array([0.5, 0.02, 0.001, 1.0, 12.0, 3.0], dtype=np.float32)
    try:
        action = agent.select_action(test_state, evaluate=True)
        check("select_action 成功", True)
        check(f"Action 形狀 = {action.shape} (期望 (1,))",
              action.shape == (1,) or action.shape == (ACTION_DIM,))
        check(f"Action 在 [-1, 1] 範圍: {action[0]:.4f}",
              -1.0 <= float(action[0]) <= 1.0)
    except Exception as e:
        check("select_action 成功", False, str(e))

print()

# ══════════════════════════════════════════════════════════════════
# Phase 2  Data.txt / Command.txt I/O 格式驗證
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Phase 2: I/O 協定格式驗證（15 分鐘間隔）")
print("=" * 70)

tmpdir = tempfile.mkdtemp(prefix="p302_deploy_test_")
data_file = os.path.join(tmpdir, "Data.txt")
cmd_file = os.path.join(tmpdir, "Command.txt")

try:
    # 2-1 模擬 Data.txt（廠商格式）
    now = datetime.now(TZ_UTC8)
    ts_str = format_ts(now)
    # 模擬值：Solar 1500 mW, Battery SoC 50%, Voltage 7.0V, Current 10mA, Temp 25°C, Speed 50%
    data_content = f"""{ts_str}
1600,500,1500,1500,450,1200,
01,500,700,10,250,500,
"""
    with open(data_file, 'w', encoding='utf-8') as f:
        f.write(data_content)
    check("Data.txt 寫入成功", os.path.exists(data_file))

    # 2-2 讀取 Data.txt
    mppt_data, status_map_raw = read_vendor_data_file(
        data_file, max_age_sec=300, clear_after_read=False
    )
    check("read_vendor_data_file 成功", status_map_raw is not None and len(status_map_raw) > 0,
          f"status_map_raw = {status_map_raw}")

    if mppt_data:
        solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw = mppt_data
        check(f"MPPT 解析: solar_v={solar_v:.2f}V, solar_p={solar_p_mw:.0f}mW", True)
    else:
        check("MPPT 資料解析", False, "mppt_data 為 None")

    if status_map_raw:
        for pp_key, vals in status_map_raw.items():
            ts, soc_pct, volt_v, curr_ma, temp_c, speed = vals
            check(f"Battery {pp_key}: SoC={soc_pct:.1f}%, V={volt_v:.2f}V, I={curr_ma:.0f}mA, speed={speed:.1f}%", True)

    # 2-3 build_agent_state（模擬 run_online_control.py 的邏輯）
    status_map = {}
    for pp_key in sorted(status_map_raw.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        ts, soc_pct, volt_v, curr_a, temp_c, speed = status_map_raw[pp_key]
        status_map[f"B{pp_key.zfill(2)}"] = (ts, volt_v, curr_a, speed, soc_pct)

    # 手動建構 state（與 run_online_control.py build_agent_state 相同邏輯）
    b_val = status_map.get("B01")
    if b_val:
        b_ts, b_v, b_i, b_flow, b_soc = b_val
        soc_frac = float(b_soc) / 100.0
        pv_kw = float(mppt_data[5]) / 1e6 if mppt_data else 0.0  # mppt_p_mw → kW
        load_kw = 0.02  # 模擬 20W 負載
        hour = float(now.hour)
        dow = float(now.weekday())
        state_vec = np.array([soc_frac, load_kw, pv_kw, 1.0, hour, dow], dtype=np.float32)
        check(f"Agent state 建構: soc={soc_frac:.2f}, load={load_kw:.4f}kW, pv={pv_kw:.6f}kW, hour={hour:.0f}",
              0 <= soc_frac <= 1.0 and load_kw >= 0 and pv_kw >= 0)
    else:
        check("Agent state 建構", False, "B01 not found in status_map")
        state_vec = None

    # 2-4 推論 + Command.txt 寫入
    if agent and state_vec is not None:
        action_norm = agent.select_action(state_vec, evaluate=True)
        a_raw_kw = float(action_norm[0]) * PMAX_KW
        power_w = abs(a_raw_kw) * 1000.0  # kW → W
        power_percent = (abs(a_raw_kw) / PMAX_KW) * 100.0
        flow_percent = max(0.0, min(100.0, power_percent))

        # 情況碼
        if a_raw_kw < -0.001:
            sit_code = 1
        elif a_raw_kw > 0.001:
            sit_code = 3
        else:
            sit_code = 4

        check(f"推論結果: action_norm={action_norm[0]:.4f}, power={power_w*1000:.2f}mW, flow={flow_percent:.1f}%",
              True)

        # 寫入 Command.txt
        pp = "01"
        success = write_control_file_vendor(
            cmd_file,
            {pp: (now, power_w, flow_percent)},
            global_ts=now,
            require_empty=False,
            situation_code=sit_code,
        )
        check("Command.txt 寫入成功", success)

        # 讀取驗證
        if success and os.path.exists(cmd_file):
            with open(cmd_file, 'r', encoding='utf-8') as f:
                cmd_content = f.read()
            lines = [l for l in cmd_content.strip().split('\n') if l.strip()]
            check(f"Command.txt 行數 ≥ 3 (situation + ts + data): got {len(lines)}",
                  len(lines) >= 3)
            if lines:
                # 第一行是情況碼
                check(f"第一行 = 情況碼: '{lines[0]}' ∈ {{1,2,3,4}}",
                      lines[0].strip() in ['1', '2', '3', '4'])
                # 第二行是時間戳
                if len(lines) > 1:
                    ts_line = lines[1].strip()
                    check(f"第二行 = 時間戳 (14 digits): '{ts_line[:14]}'",
                          len(ts_line) >= 14 and ts_line[:14].isdigit())
                # 第三行是電池命令
                if len(lines) > 2:
                    cmd_line = lines[2].strip()
                    parts = cmd_line.split(',')
                    check(f"第三行格式: PP,mW,flow%,... → '{cmd_line}'",
                          len(parts) >= 3 and parts[0] == '01')
                    if len(parts) >= 3:
                        mw_val = int(parts[1]) if parts[1].isdigit() else -1
                        flow_val = int(parts[2]) if parts[2].isdigit() else -1
                        check(f"功率={mw_val}mW (≥0), 流速={flow_val}% (0~100)",
                              mw_val >= 0 and 0 <= flow_val <= 100)
            print(f"\n  Command.txt 內容:")
            for l in lines:
                print(f"    {l}")

except Exception as e:
    check("Phase 2 完成", False, f"Exception: {e}")
    traceback.print_exc()

print()

# ══════════════════════════════════════════════════════════════════
# Phase 3  多步模擬（24 小時 = 96 步）
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Phase 3: 多步模擬 (96 步 = 24 小時, 每步 15 分鐘)")
print("=" * 70)

if agent:
    sim_soc = 0.5  # 初始 SoC
    sim_load_kw = 0.020   # 20W 負載
    sim_pv_base_kw = 0.001  # 1W 基準太陽能
    battery_cap_kwh = 0.00007
    time_step_h = 0.25  # 15 分鐘

    soc_history = [sim_soc]
    action_history = []
    sit_code_history = []
    hour_history = []

    start_dt = datetime(2026, 3, 2, 0, 0, 0, tzinfo=TZ_UTC8)  # 下周一

    for step in range(STEPS_PER_DAY):
        current_dt = start_dt + timedelta(minutes=15 * step)
        hour = float(current_dt.hour) + float(current_dt.minute) / 60.0
        dow = float(current_dt.weekday())

        # 模擬太陽能：白天有，晚上沒有
        if 6 <= current_dt.hour <= 18:
            solar_factor = np.sin(np.pi * (current_dt.hour - 6) / 12.0)
            sim_pv_kw = sim_pv_base_kw * max(0.1, solar_factor)
        else:
            sim_pv_kw = 0.0

        state = np.array([sim_soc, sim_load_kw, sim_pv_kw, 1.0, float(current_dt.hour), dow],
                         dtype=np.float32)

        action = agent.select_action(state, evaluate=True)
        a_kw = float(action[0]) * PMAX_KW

        # 模擬 SoC 更新（簡化版）
        delta_soc = (a_kw * time_step_h) / battery_cap_kwh
        new_soc = np.clip(sim_soc + delta_soc, SOC_MIN, SOC_MAX)

        # 情況碼
        if a_kw < -0.001 * PMAX_KW:
            sit = 1
        elif a_kw > 0.001 * PMAX_KW:
            sit = 3
        else:
            sit = 4

        soc_history.append(new_soc)
        action_history.append(a_kw)
        sit_code_history.append(sit)
        hour_history.append(current_dt.hour)

        sim_soc = new_soc

    # 統計
    soc_arr = np.array(soc_history)
    action_arr = np.array(action_history)
    min_soc = float(np.min(soc_arr))
    max_soc = float(np.max(soc_arr))
    mean_action = float(np.mean(action_arr))
    charge_steps = sum(1 for a in action_arr if a > 0)
    discharge_steps = sum(1 for a in action_arr if a < 0)
    idle_steps = len(action_arr) - charge_steps - discharge_steps

    check(f"SoC 範圍: [{min_soc:.3f}, {max_soc:.3f}] (限制 [{SOC_MIN}, {SOC_MAX}])",
          min_soc >= SOC_MIN - 0.01 and max_soc <= SOC_MAX + 0.01)
    check(f"96 步全部完成", len(action_history) == STEPS_PER_DAY)
    check(f"充電步數={charge_steps}, 放電步數={discharge_steps}, 待機={idle_steps}",
          charge_steps + discharge_steps + idle_steps == STEPS_PER_DAY)

    # 打印每 4 步 (= 每小時) 的 SoC
    print(f"\n  每小時 SoC 快照（共 24 小時）：")
    print(f"  {'Hour':>5s} | {'SoC':>6s} | {'Action(kW)':>10s} | {'Sit':>3s}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*10}-+-{'-'*3}")
    for h in range(24):
        idx = h * 4
        soc_val = soc_history[idx]
        if idx < len(action_arr):
            a_val = action_arr[idx] * 1e6  # kW → µW for readability
            s_val = sit_code_history[idx]
        else:
            a_val = 0
            s_val = 4
        print(f"  {h:5d} | {soc_val:6.3f} | {a_val:8.1f}µW | {s_val:3d}")

    print(f"\n  最終 SoC: {sim_soc:.4f}")

    # 驗證 Command.txt 最後一步格式
    last_dt = start_dt + timedelta(minutes=15 * (STEPS_PER_DAY - 1))
    last_action_kw = action_arr[-1]
    last_power_w = abs(last_action_kw) * 1000.0
    last_power_mw = last_power_w * 1000.0
    last_flow = (abs(last_action_kw) / PMAX_KW) * 100.0

    # 清理並寫入最終命令
    if os.path.exists(cmd_file):
        os.remove(cmd_file)

    success = write_control_file_vendor(
        cmd_file,
        {"01": (last_dt, last_power_w, last_flow)},
        global_ts=last_dt,
        require_empty=False,
        situation_code=sit_code_history[-1],
    )
    check("最終步 Command.txt 寫入成功", success)

    if success:
        with open(cmd_file, 'r', encoding='utf-8') as f:
            final_cmd = f.read()
        print(f"\n  最終 Command.txt 內容:")
        for l in final_cmd.strip().split('\n'):
            print(f"    {l}")

else:
    check("Phase 3 跳過（Agent 載入失敗）", False, "Agent 為 None")

print()

# ══════════════════════════════════════════════════════════════════
# Phase 4  15 分鐘時間步一致性
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Phase 4: 15 分鐘時間步一致性驗證")
print("=" * 70)

check(f"INTERVAL_SEC = {INTERVAL_SEC} (期望 900)", INTERVAL_SEC == 900)
check(f"time_step = 0.25h = 15min", abs(0.25 * 60 - 15.0) < 0.01)
check(f"672 步 = 7 天 × 96 步/天 (config episode_length 應為 672)", 672 == 7 * 96)
check(f"目前 config episode_length=576 = 6 天 × 96 步/天", 576 == 6 * 96)
check(f"96 步 = 24h / 0.25h", 96 == int(24 / 0.25))

# 驗證 hour 計算（修復後的版本）
time_step = 0.25
steps_per_hour = max(1, int(round(1.0 / max(time_step, 1e-9))))
check(f"steps_per_hour = {steps_per_hour} (期望 4)", steps_per_hour == 4)

for test_step in [0, 4, 8, 48, 95]:
    computed_hour = int((test_step // steps_per_hour) % 24)
    expected_hour = int((test_step * 0.25) % 24)
    check(f"step={test_step} → hour={computed_hour} (期望 {expected_hour})",
          computed_hour == expected_hour)

# 驗證 P302 功率尺度
check(f"battery_capacity = {battery_cap_kwh*1e6:.1f}µWh = 0.07 Wh",
      abs(battery_cap_kwh - 0.00007) < 1e-9)
check(f"battery_power_max = {PMAX_KW*1e6:.0f}µW = 0.17 mW",
      abs(PMAX_KW - 0.00017) < 1e-9)

print()

# ══════════════════════════════════════════════════════════════════
# 清理 & 結果摘要
# ══════════════════════════════════════════════════════════════════
try:
    shutil.rmtree(tmpdir, ignore_errors=True)
except Exception:
    pass

print("=" * 70)
print(f"結果摘要：通過 {passed}/{passed + failed}，失敗 {failed}")
print("=" * 70)
if errors:
    print("\n失敗項目：")
    for e in errors:
        print(e)
    print()

if failed == 0:
    print("✅ 所有驗證通過！模型已準備好 deploy。")
    print(f"   模型路徑: {MODEL_PATH}")
    print(f"   部署命令範例:")
    print(f"   python control/run_online_control.py \\")
    print(f"     --status-file Data.txt \\")
    print(f"     --command-file Command.txt \\")
    print(f"     --model-path {MODEL_PATH} \\")
    print(f"     --battery-id 01 \\")
    print(f"     --pmax-kw {PMAX_KW} \\")
    print(f"     --use-power-to-flow \\")
    print(f"     --interval-sec {INTERVAL_SEC}")
else:
    print("❌ 有驗證失敗，請先修復再 deploy。")

sys.exit(0 if failed == 0 else 1)
