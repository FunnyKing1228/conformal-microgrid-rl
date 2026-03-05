#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solar-Only Data Collection Script (Scenario 4 - Standby)
=========================================================
Purpose:
  - Collect MPPT / solar data without using the battery
  - Battery power = 0, flow = 0
  - Load groups = 4 (constant, via load_count)
  - Scenario 4 (Standby): grid covers 100% of load
  - Logs all MPPT readings to CSV for training data

Command.txt output format:
  4                          <-- Scenario 4 (standby)
  YYYYMMDDhhmmss,4           <-- timestamp, load_count=4
  01,0,0,                    <-- PP=01, power=0mW, flow=0%

Usage (on deployment machine):
  1. Place this script next to Data.txt / Command.txt
     (or specify paths via --data-file / --command-file)
  2. Run: python solar_test_collect.py
  3. Press Ctrl+C to stop. CSV log saved to ./solar_log_YYYYMMDD_HHMMSS.csv

No dependencies beyond Python standard library + (optional) numpy for stats.
"""

import os
import sys
import io
import csv
import time
import shutil
import tempfile
import argparse
from datetime import datetime, timezone, timedelta

# Optional: numpy for better stats
try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

TZ_UTC8 = timezone(timedelta(hours=8))


# ====================================================================
# Minimal Data.txt parser (standalone, no external dependencies)
# ====================================================================
def parse_data_txt(path):
    """
    Parse vendor Data.txt format (supports both old 6-field and new 9-field MPPT).
    
    New format (2026/03):
        Line 1: YYYYMMDDHHmmSS
        Line 2: SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,BusV,BusI,BusP,
        Line 3: LoadV,LoadI,LoadP,
        Line 4+: ID,SOC,BV,BI,Temp,Speed,
    
    Old format:
        Line 1: YYYYMMDDHHmmSS
        Line 2: SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,
        Line 3+: ID,SOC,BV,BI,Temp,Speed,
    
    Returns: (timestamp_str, mppt_dict, mppt_bus_dict, load_dict, battery_list)
    
    mppt_dict keys:     solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw
    mppt_bus_dict keys: bus_v, bus_i_ma, bus_p_mw  (or None if old format)
    load_dict keys:     load_v, load_i_ma, load_p_mw  (or None if old format)
    battery_list: [{pp, soc_pct, volt_v, curr_ma, temp_c, speed_pct}, ...]
    """
    if not os.path.exists(path):
        return None, None, None, None, []
    try:
        with io.open(path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return None, None, None, None, []

    if not lines:
        return None, None, None, None, []

    # Line 1: timestamp (14 digits, optionally followed by ,load_count)
    ts_str = None
    idx = 0
    first = lines[0]
    ts_part = first.split(',')[0].strip()
    if len(ts_part) >= 14 and ts_part[:14].isdigit():
        ts_str = ts_part[:14]
        idx = 1

    # Line 2: MPPT data (6 or 9 numeric fields, first field is NOT a small battery ID)
    mppt = None
    mppt_bus = None
    if idx < len(lines):
        parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
        if len(parts) >= 6:
            is_battery_id = parts[0].isdigit() and 1 <= int(parts[0]) <= 10
            if not is_battery_id:
                try:
                    vals = [float(p) for p in parts]
                    mppt = {
                        'solar_v': vals[0] / 100.0,
                        'solar_i_ma': vals[1],
                        'solar_p_mw': vals[2],
                        'mppt_v': vals[3] / 100.0,
                        'mppt_i_ma': vals[4],
                        'mppt_p_mw': vals[5],
                    }
                    # New format: 9+ fields → MPPT-Bus
                    if len(vals) >= 9:
                        mppt_bus = {
                            'bus_v': vals[6] / 100.0,
                            'bus_i_ma': vals[7],
                            'bus_p_mw': vals[8],
                        }
                    idx += 1
                except (ValueError, IndexError):
                    idx += 1

    # Line 3 (new format only): Load power data (3 fields, only if MPPT-Bus was detected)
    load = None
    if idx < len(lines) and mppt_bus is not None:
        parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
        if len(parts) >= 3:
            is_battery = (parts[0].isdigit() and 1 <= int(parts[0]) <= 10
                          and len(parts) >= 6)
            if not is_battery:
                try:
                    vals = [float(p) for p in parts[:3]]
                    load = {
                        'load_v': vals[0] / 100.0,    # 0.01V
                        'load_i_ma': vals[1],           # mA
                        'load_p_mw': vals[2],           # mW
                    }
                    idx += 1
                except (ValueError, IndexError):
                    idx += 1

    # Remaining lines: battery data
    batteries = []
    while idx < len(lines):
        parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
        idx += 1
        if len(parts) < 6:
            continue
        try:
            batteries.append({
                'pp': parts[0],
                'soc_pct': float(parts[1]) / 10.0,
                'volt_v': float(parts[2]) / 100.0,
                'curr_ma': float(parts[3]),
                'temp_c': float(parts[4]) / 10.0,
                'speed_pct': float(parts[5]) / 10.0,
            })
        except (ValueError, IndexError):
            continue

    return ts_str, mppt, mppt_bus, load, batteries


# ====================================================================
# Minimal Command.txt writer (standalone)
# ====================================================================
def write_command_txt(path, scenario, timestamp_dt, load_count, pp, power_mw, flow_pct):
    """
    Write Command.txt in vendor format:
      {scenario}
      YYYYMMDDhhmmss,{load_count}
      PP,power_mW,flow_pct,
    
    Uses atomic write (temp file + move) for safety.
    """
    ts_str = timestamp_dt.strftime('%Y%m%d%H%M%S')
    content = (
        f"{scenario}\n"
        f"{ts_str},{load_count}\n"
        f"{pp},{int(power_mw)},{int(flow_pct)},\n"
    )
    
    dir_name = os.path.dirname(path) or '.'
    try:
        fd, temp_path = tempfile.mkstemp(
            prefix=os.path.basename(path) + '.',
            suffix='.tmp',
            dir=dir_name,
            text=True
        )
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()
        time.sleep(0.01)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        shutil.move(temp_path, path)
        return True
    except Exception as e:
        # Fallback: direct write
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        try:
            with io.open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception:
            return False


# ====================================================================
# Main loop
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Solar-only data collection (Scenario 4 - Standby)')
    parser.add_argument('--data-file', type=str, default='./Data.txt',
                        help='Data.txt path (vendor writes)')
    parser.add_argument('--command-file', type=str, default='./Command.txt',
                        help='Command.txt path (we write)')
    parser.add_argument('--battery-pp', type=str, default='01',
                        help='Battery PP ID (default: 01)')
    parser.add_argument('--load-count', type=int, default=4,
                        help='Load groups (0-4, default: 4)')
    parser.add_argument('--poll-sec', type=float, default=10.0,
                        help='Read Data.txt + write CSV interval (sec, default: 10)')
    parser.add_argument('--log-dir', type=str, default='.',
                        help='CSV log output directory')
    parser.add_argument('--scenario', type=int, default=4, choices=[1, 2, 3, 4],
                        help='Scenario code (default: 4=Standby)')
    args = parser.parse_args()

    # CSV 格式與舊版 DataCollector 相容 + 新增 MPPT-Bus / Load 欄位
    CSV_HEADER = [
        'timestamp',           # 時間戳
        'battery_id',          # 電池 ID
        'soc_percent',         # SoC (%)
        'voltage_v',           # 電壓 (V)
        'current_ma',          # 電流 (mA)
        'temp_c',              # 溫度 (°C)
        'speed_percent',       # 流速 (%)
        'solar_v',             # 太陽能電壓 (V)
        'solar_i_ma',          # 太陽能電流 (mA)
        'solar_p_mw',          # 太陽能功率 (mW)
        'mppt_v',              # MPPT 電壓 (V)
        'mppt_i_ma',           # MPPT 電流 (mA)
        'mppt_p_mw',           # MPPT 功率 (mW)
        'bus_v',               # MPPT-Bus 電壓 (V)   ← 新增
        'bus_i_ma',            # MPPT-Bus 電流 (mA)  ← 新增
        'bus_p_mw',            # MPPT-Bus 功率 (mW)  ← 新增
        'load_v',              # 負載電壓 (V)        ← 新增
        'load_i_ma',           # 負載電流 (mA)       ← 新增
        'load_p_mw',           # 負載功率 (mW)       ← 新增
        'load_count',          # 負載組數
        'load_power_w',        # 負載總功率 (W, 估計值)
        'data_txt_ts',         # Data.txt 原始時間戳
        'elapsed_sec',         # 累計秒數
    ]

    os.makedirs(args.log_dir, exist_ok=True)

    # --- 依日期開檔：存在就 append，不存在就新建 + 寫 header ---
    def open_csv_for_date(date_str):
        """開啟或建立指定日期的 CSV，回傳 (file, writer, path)"""
        path = os.path.join(args.log_dir, f'collected_data_{date_str}.csv')
        if os.path.exists(path):
            f = open(path, 'a', newline='', encoding='utf-8-sig')
            w = csv.DictWriter(f, fieldnames=CSV_HEADER)
            print(f'  [CSV] 接續寫入: {path}')
        else:
            f = open(path, 'w', newline='', encoding='utf-8-sig')
            w = csv.DictWriter(f, fieldnames=CSV_HEADER)
            w.writeheader()
            print(f'  [CSV] 建立新檔: {path}')
        return f, w, path

    current_date_str = datetime.now(TZ_UTC8).strftime('%Y-%m-%d')
    csv_file, csv_writer, csv_path = open_csv_for_date(current_date_str)

    pp = f'{int(args.battery_pp):02d}'
    total_reads = 0
    start_time = time.time()

    print('=' * 70)
    print('  P302 Solar-Only Data Collection')
    print('=' * 70)
    print(f'  Scenario  : {args.scenario} ({"Standby" if args.scenario == 4 else "S" + str(args.scenario)})')
    print(f'  Load      : {args.load_count} groups ({args.load_count * 12}W)')
    print(f'  Battery   : PP={pp}, Power=0mW, Flow=0%')
    print(f'  Data.txt  : {os.path.abspath(args.data_file)}')
    print(f'  Command   : {os.path.abspath(args.command_file)}')
    print(f'  CSV Log   : {os.path.abspath(csv_path)}')
    print(f'  Poll      : {args.poll_sec}s')
    print('=' * 70)
    print()
    print('  Press Ctrl+C to stop.')
    print()
    print('-' * 70)

    # Running stats
    mppt_values = []

    try:
        while True:
            loop_start = time.time()
            now = datetime.now(TZ_UTC8)

            # 跨日檢查：日期變了就開新 CSV
            today_str = now.strftime('%Y-%m-%d')
            if today_str != current_date_str:
                csv_file.close()
                current_date_str = today_str
                csv_file, csv_writer, csv_path = open_csv_for_date(current_date_str)

            # 1) Write Command.txt (scenario 4, power=0, flow=0)
            ok = write_command_txt(
                args.command_file,
                scenario=args.scenario,
                timestamp_dt=now,
                load_count=args.load_count,
                pp=pp,
                power_mw=0,
                flow_pct=0,
            )

            # 2) Read Data.txt (supports old 6-field and new 9-field + load)
            ts_str, mppt, mppt_bus, load, batteries = parse_data_txt(args.data_file)
            elapsed = time.time() - start_time

            # 負載功率：優先用 Data.txt 實測值，否則用估計值
            if load is not None:
                load_power_w = load['load_p_mw'] / 1000.0  # mW → W
            else:
                load_power_w = args.load_count * 12.0  # 每組 12W @5V (估計)

            row = {
                'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                'battery_id': pp,
                'soc_percent': '',
                'voltage_v': '',
                'current_ma': '',
                'temp_c': '',
                'speed_percent': '',
                'solar_v': '',
                'solar_i_ma': '',
                'solar_p_mw': '',
                'mppt_v': '',
                'mppt_i_ma': '',
                'mppt_p_mw': '',
                'bus_v': '',
                'bus_i_ma': '',
                'bus_p_mw': '',
                'load_v': '',
                'load_i_ma': '',
                'load_p_mw': '',
                'load_count': str(args.load_count),
                'load_power_w': f'{load_power_w:.1f}',
                'data_txt_ts': ts_str or '',
                'elapsed_sec': f'{elapsed:.1f}',
            }

            if mppt:
                row.update({
                    'solar_v': f'{mppt["solar_v"]:.2f}',
                    'solar_i_ma': f'{mppt["solar_i_ma"]:.0f}',
                    'solar_p_mw': f'{mppt["solar_p_mw"]:.0f}',
                    'mppt_v': f'{mppt["mppt_v"]:.2f}',
                    'mppt_i_ma': f'{mppt["mppt_i_ma"]:.0f}',
                    'mppt_p_mw': f'{mppt["mppt_p_mw"]:.0f}',
                })
                mppt_values.append(mppt['mppt_p_mw'])
                total_reads += 1

            if mppt_bus:
                row.update({
                    'bus_v': f'{mppt_bus["bus_v"]:.2f}',
                    'bus_i_ma': f'{mppt_bus["bus_i_ma"]:.0f}',
                    'bus_p_mw': f'{mppt_bus["bus_p_mw"]:.0f}',
                })

            if load:
                row.update({
                    'load_v': f'{load["load_v"]:.2f}',
                    'load_i_ma': f'{load["load_i_ma"]:.0f}',
                    'load_p_mw': f'{load["load_p_mw"]:.0f}',
                })

            # Battery info (if available)
            batt = None
            for b in batteries:
                if b['pp'] == pp or b['pp'] == args.battery_pp:
                    batt = b
                    break
            if batt:
                row.update({
                    'soc_percent': f'{batt["soc_pct"]:.2f}',
                    'voltage_v': f'{batt["volt_v"]:.2f}',
                    'current_ma': f'{batt["curr_ma"]:.0f}',
                    'temp_c': f'{batt["temp_c"]:.1f}',
                    'speed_percent': f'{batt["speed_pct"]:.1f}',
                })

            csv_writer.writerow(row)
            csv_file.flush()

            # Console output (every read)
            ts_display = now.strftime('%H:%M:%S')
            if mppt:
                mppt_w = mppt['mppt_p_mw'] / 1000.0
                solar_w = mppt['solar_p_mw'] / 1000.0
                
                extra = ''
                if mppt_bus:
                    bus_w = mppt_bus['bus_p_mw'] / 1000.0
                    extra += f'  Bus={bus_w:.3f}W'
                if load:
                    load_w = load['load_p_mw'] / 1000.0
                    extra += f'  Load={load_w:.3f}W'
                
                batt_info = ''
                if batt:
                    batt_info = (f'  Batt: {batt["volt_v"]:.2f}V '
                                 f'{batt["curr_ma"]:.0f}mA '
                                 f'SoC={batt["soc_pct"]:.1f}% '
                                 f'T={batt["temp_c"]:.1f}C')

                # Stats
                stats = ''
                if len(mppt_values) > 5:
                    recent = mppt_values[-30:]
                    avg = sum(recent) / len(recent)
                    if HAS_NP:
                        std = float(np.std(recent))
                        stats = f'  [avg={avg:.0f} std={std:.0f}]'
                    else:
                        stats = f'  [avg={avg:.0f}]'

                cmd_ok = 'OK' if ok else 'FAIL'
                print(f'  [{ts_display}] #{total_reads:5d}  '
                      f'MPPT={mppt["mppt_p_mw"]:6.0f}mW ({mppt_w:.3f}W)  '
                      f'Solar={solar_w:.3f}W'
                      f'{extra}'
                      f'{batt_info}{stats}  [Cmd:{cmd_ok}]')
            else:
                print(f'  [{ts_display}] No MPPT data (waiting for Data.txt...)'
                      f'  [Cmd:{"OK" if ok else "FAIL"}]')

            # Summary every 5 minutes
            if total_reads > 0 and total_reads % 30 == 0:
                mins = elapsed / 60.0
                rate = total_reads / mins if mins > 0 else 0
                print(f'\n  --- {mins:.1f} min | {total_reads} reads '
                      f'({rate:.1f}/min) | CSV: {csv_path} ---\n')

            # Sleep
            loop_elapsed = time.time() - loop_start
            sleep_time = max(0.1, args.poll_sec - loop_elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f'\n\n{"=" * 70}')
        print(f'  Data collection stopped (Ctrl+C)')
        print(f'  Duration  : {elapsed/60:.1f} minutes')
        print(f'  Readings  : {total_reads}')
        if mppt_values:
            avg_mw = sum(mppt_values) / len(mppt_values)
            max_mw = max(mppt_values)
            print(f'  MPPT avg  : {avg_mw:.0f} mW ({avg_mw/1000:.3f} W)')
            print(f'  MPPT max  : {max_mw:.0f} mW ({max_mw/1000:.3f} W)')
        print(f'  CSV saved : {os.path.abspath(csv_path)}')
        print(f'{"=" * 70}')

        # Write standby command one last time
        now = datetime.now(TZ_UTC8)
        write_command_txt(
            args.command_file,
            scenario=4,
            timestamp_dt=now,
            load_count=args.load_count,
            pp=pp,
            power_mw=0,
            flow_pct=0,
        )

    finally:
        csv_file.close()


if __name__ == '__main__':
    main()
