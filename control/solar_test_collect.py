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
    Parse vendor Data.txt format.
    Returns: (timestamp_str, mppt_dict, battery_list)
    
    mppt_dict keys: solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw
    battery_list: [{pp, soc_pct, volt_v, curr_ma, temp_c, speed_pct}, ...]
    """
    if not os.path.exists(path):
        return None, None, []
    try:
        with io.open(path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return None, None, []

    if not lines:
        return None, None, []

    # Line 1: timestamp (14 digits, optionally followed by ,load_count)
    ts_str = None
    idx = 0
    first = lines[0]
    ts_part = first.split(',')[0].strip()
    if len(ts_part) >= 14 and ts_part[:14].isdigit():
        ts_str = ts_part[:14]
        idx = 1

    # Line 2: MPPT data (6 numeric fields, first field is NOT a small battery ID)
    mppt = None
    if idx < len(lines):
        parts = [p.strip() for p in lines[idx].split(',') if p.strip()]
        if len(parts) >= 6:
            # Check if first field could be a battery ID (1-10)
            is_battery_id = parts[0].isdigit() and 1 <= int(parts[0]) <= 10
            if not is_battery_id:
                try:
                    vals = [float(p) for p in parts[:6]]
                    mppt = {
                        'solar_v': vals[0] / 100.0,       # 0.01V
                        'solar_i_ma': vals[1],              # 1mA
                        'solar_p_mw': vals[2],              # 1mW
                        'mppt_v': vals[3] / 100.0,         # 0.01V
                        'mppt_i_ma': vals[4],               # 1mA
                        'mppt_p_mw': vals[5],               # 1mW
                    }
                    idx += 1
                except (ValueError, IndexError):
                    idx += 1  # skip bad line

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
                'soc_pct': float(parts[1]) / 10.0,     # 0.1%
                'volt_v': float(parts[2]) / 100.0,      # 0.01V
                'curr_ma': float(parts[3]),               # 1mA
                'temp_c': float(parts[4]) / 10.0,        # 0.1C
                'speed_pct': float(parts[5]) / 10.0,     # 0.1%
            })
        except (ValueError, IndexError):
            continue

    return ts_str, mppt, batteries


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
    parser.add_argument('--poll-sec', type=float, default=1.0,
                        help='Command.txt update interval (sec, default: 1)')
    parser.add_argument('--log-dir', type=str, default='.',
                        help='CSV log output directory')
    parser.add_argument('--scenario', type=int, default=4, choices=[1, 2, 3, 4],
                        help='Scenario code (default: 4=Standby)')
    args = parser.parse_args()

    # CSV log
    os.makedirs(args.log_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(args.log_dir, f'collected_data_{date_str}.csv')

    # CSV 格式與舊版 DataCollector 相容
    csv_header = [
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
        'load_count',          # 負載組數
        'load_power_w',        # 負載總功率 (W)
        'data_txt_ts',         # Data.txt 原始時間戳 (bonus)
        'elapsed_sec',         # 累計秒數 (bonus)
    ]
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8-sig')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_header)
    csv_writer.writeheader()

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

            # 2) Read Data.txt
            ts_str, mppt, batteries = parse_data_txt(args.data_file)
            elapsed = time.time() - start_time

            load_power_w = args.load_count * 12.0  # 每組 12W @5V

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
                batt_info = ''
                if batt:
                    batt_info = (f'  Batt: {batt["volt_v"]:.2f}V '
                                 f'{batt["curr_ma"]:.0f}mA '
                                 f'SoC={batt["soc_pct"]:.1f}% '
                                 f'T={batt["temp_c"]:.1f}C')

                # Stats
                stats = ''
                if len(mppt_values) > 5:
                    recent = mppt_values[-30:]  # last 30
                    avg = sum(recent) / len(recent)
                    if HAS_NP:
                        std = float(np.std(recent))
                        stats = f'  [avg={avg:.0f} std={std:.0f}]'
                    else:
                        stats = f'  [avg={avg:.0f}]'

                cmd_ok = 'OK' if ok else 'FAIL'
                print(f'  [{ts_display}] #{total_reads:5d}  '
                      f'MPPT={mppt["mppt_p_mw"]:6.0f}mW ({mppt_w:.3f}W)  '
                      f'Solar={solar_w:.3f}W  '
                      f'V={mppt["mppt_v"]:.2f}V  I={mppt["mppt_i_ma"]:.0f}mA'
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
