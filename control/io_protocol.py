import os
import io
import time
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, Optional, List

# 解析與格式化輔助（根據需求：固定 8 碼，前 1 碼為符號位或固定 0，後 7 碼兩位小數）

TZ_UTC8 = timezone(timedelta(hours=8))


def parse_ts(ts_str: str) -> datetime:
	"""YYYYMMDDhhmmss（UTC+8）"""
	ts_str = ts_str.strip()
	dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
	return dt.replace(tzinfo=TZ_UTC8)


def format_ts(dt: Optional[datetime] = None) -> str:
	if dt is None:
		dt = datetime.now(TZ_UTC8)
	else:
		if dt.tzinfo is None:
			dt = dt.replace(tzinfo=TZ_UTC8)
		dt = dt.astimezone(TZ_UTC8)
	return dt.strftime("%Y%m%d%H%M%S")


def _to_7digits(value: float) -> str:
	iv = int(round(abs(value) * 100.0))
	return f"{iv:07d}"[-7:]


def parse_signed_field(s: str) -> float:
	"""通用有符號欄位：第1碼 0=正/1=負；後7碼兩位小數。"""
	s = s.strip()
	if len(s) != 8 or not s[0] in ("0", "1") or not s[1:].isdigit():
		raise ValueError(f"Invalid signed field: {s!r}")
	mag = int(s[1:]) / 100.0
	return mag if s[0] == "0" else -mag


def format_signed_field(value: float) -> str:
	sign = "0" if value >= 0 else "1"
	return f"{sign}{_to_7digits(value)}"


def parse_unsigned_field(s: str) -> float:
	"""通用無符號欄位：第1碼固定 0；後7碼兩位小數。"""
	s = s.strip()
	if len(s) != 8 or s[0] != "0" or not s[1:].isdigit():
		raise ValueError(f"Invalid unsigned field: {s!r}")
	return int(s[1:]) / 100.0


def format_unsigned_field(value: float) -> str:
	return f"0{_to_7digits(max(0.0, value))}"


# ---- 模型 → 元件（命令）----
# 行格式：PP,YYYYMMDDhhmmss,SMMMMMMM,FMMMMMMM,RES1,RES2,RES3,RES4,RES5,RES6
# PP = "01"..."99"

def parse_control_line(line: str) -> Tuple[str, datetime, float, float]:
	parts = [p.strip() for p in line.strip().split(",")]
	if len(parts) < 4:
		raise ValueError(f"Invalid control line: {line!r}")
	pp = parts[0]
	ts = parse_ts(parts[1])
	s_power_w = parse_signed_field(parts[2])     # W
	f_flow_ccm = parse_unsigned_field(parts[3])  # cc/min
	return pp, ts, s_power_w, f_flow_ccm


def format_control_line(pp: str, ts: datetime, power_w: float, flow_ccm: float) -> str:
	pp2 = f"{int(pp):02d}" if pp.isdigit() else pp
	return f"{pp2},{format_ts(ts)},{format_signed_field(power_w)},{format_unsigned_field(flow_ccm)}, , , , , , "


def read_control_file(path: str, max_age_sec: Optional[int] = None) -> Dict[str, Tuple[datetime, float, float]]:
	"""讀取命令文件，回傳每個 PP 的最新一筆。讀完清空檔案。"""
	results: Dict[str, Tuple[datetime, float, float]] = {}
	try:
		if not os.path.exists(path):
			return results
		with io.open(path, "r", encoding="utf-8") as f:
			lines = [ln for ln in f.readlines() if ln.strip()]
	except Exception:
		return results
	finally:
		# 清空檔案避免重複讀取
		try:
			with io.open(path, "w", encoding="utf-8") as w:
				w.write("")
		except Exception:
			pass

	now_ts = datetime.now(TZ_UTC8)
	for ln in lines:
		try:
			pp, ts, p_w, f_ccm = parse_control_line(ln)
		except Exception:
			continue
		if max_age_sec is not None and (now_ts - ts).total_seconds() > float(max_age_sec):
			continue
		if pp not in results or ts > results[pp][0]:
			results[pp] = (ts, p_w, f_ccm)
	return results


def write_control_file(path: str, commands: Dict[str, Tuple[datetime, float, float]]) -> None:
	"""覆寫命令文件（一次性寫入多顆電池指令）。使用舊格式（每行含時間戳）。"""
	lines: List[str] = []
	for pp, (ts, power_w, flow_ccm) in commands.items():
		lines.append(format_control_line(pp, ts, power_w, flow_ccm))
	out = "\n".join(lines) + ("\n" if lines else "")
	with io.open(path, "w", encoding="utf-8") as f:
		f.write(out)


def check_file_available(path: str, max_wait_sec: float = 0.1, check_empty: bool = True) -> bool:
	"""
	檢查檔案是否可用（空或可控制）。
	
	重要：廠商程式會每秒開一次檔案，因此需要確認檔案狀態才能接管控制。
	
	Args:
		path: 檔案路徑
		max_wait_sec: 最大等待時間（秒），用於重試
		check_empty: 是否檢查檔案為空（True=必須為空，False=只檢查可寫入）
	
	Returns:
		True 如果檔案可用（空或不存在），False 如果被占用或非空
	"""
	# 如果檔案不存在，視為可用
	if not os.path.exists(path):
		return True
	
	# 檢查檔案是否為空
	if check_empty:
		try:
			# 快速檢查檔案大小
			if os.path.getsize(path) > 0:
				# 檔案非空，嘗試讀取確認
				with io.open(path, "r", encoding="utf-8") as f:
					content = f.read().strip()
					if content:
						return False  # 檔案有內容，不可用
		except (IOError, OSError):
			# 檔案被占用，不可用
			return False
	
	# 嘗試以寫入模式開啟（測試是否可控制）
	# 在 Windows 上，如果檔案被其他程式開啟，可能會失敗
	try:
		# 使用臨時模式測試寫入權限
		with io.open(path, "r+", encoding="utf-8") as f:
			pass
		return True
	except (IOError, OSError, PermissionError):
		# 檔案被占用或無權限
		return False


def write_control_file_vendor(path: str, commands: Dict[str, Tuple[datetime, float, float]], 
                              global_ts: Optional[datetime] = None,
                              require_empty: bool = True,
                              max_wait_sec: float = 0.1,
                              max_retries: int = 3,
                              load_count: Optional[int] = None,
                              situation_code: Optional[int] = None) -> bool:
	"""
	覆寫命令文件（廠商格式：第一行時間戳，後面每行無時間戳）。
	
	重要：廠商程式會每秒開一次檔案，因此需要先確認檔案是空的或可控制才能寫入。
	
	格式（有 situation_code 時）：
		{situation_code}         ← 1~4 其中一個（能流情況碼）
		YYYYMMDDhhmmss
		01,功率(mW),流速,
		...
	
	情況碼定義（前提：太陽能僅供負載，不逆送；剩餘負載 = 負載 - 太陽能）：
		1 = 電池放電全包：電池獨立滿足剩餘負載（無市電）
		2 = 電池放電＋市電補足：電池放電不足，差額由市電補
		3 = 市電同時滿足負載及充電：電池充電，市電輸出 = 充電功率 + 剩餘負載
		4 = 電池待機：100% 由市電滿足剩餘負載
	
	格式說明：
		- 功率：mW，不補0（8000 = 8000 mW = 8 kW）
		- 流速：百分比，不補0，1~3位數都可以（4 = 4%, 25 = 25%, 100 = 100%）
	
	注意：「command比不足:9」是指電池數量不足，必須寫入所有電池（01-10），未使用的電池也要寫 0
	
	Args:
		path: 輸出檔案路徑（應為 command.txt）
		commands: {PP: (ts, power_w, flow_percent)}，ts 僅用於驗證，實際使用 global_ts
		global_ts: 全域時間戳（第一行），若為 None 則使用 commands 中最早的 ts 或當前時間
		require_empty: 是否要求檔案為空才能寫入（True=必須為空，False=只檢查可寫入）
		max_wait_sec: 每次重試前的等待時間（秒）
		max_retries: 最大重試次數
		situation_code: 能流情況碼（1~4），若提供則寫在第一行（時間戳之前）
	
	Returns:
		True 如果成功寫入，False 如果檔案不可用或寫入失敗
	"""
	# 檢查檔案是否可用（重試機制）
	for attempt in range(max_retries):
		if check_file_available(path, max_wait_sec=max_wait_sec, check_empty=require_empty):
			break
		if attempt < max_retries - 1:
			time.sleep(max_wait_sec)
		else:
			# 所有重試都失敗
			return False
	
	# 準備內容
	if not commands:
		# 空指令時，只寫時間戳
		ts_line = format_ts(global_ts) if global_ts else format_ts()
		# 如果有負載顆數，加上它
		if load_count is not None:
			ts_line = f"{ts_line},{load_count}"
		# 如果有情況碼，加在最前面
		if situation_code is not None:
			content = f"{int(situation_code)}\n{ts_line}\n"
		else:
			content = ts_line + "\n"
	else:
		# 決定全域時間戳
		if global_ts is None:
			# 使用最早的 ts 或當前時間
			all_ts = [ts for ts, _, _ in commands.values()]
			if all_ts:
				global_ts = min(all_ts)
			else:
				global_ts = datetime.now(TZ_UTC8)
		
		# 時間戳行（如果 ID=0 或指定了負載顆數，加上負載顆數）
		ts_line = format_ts(global_ts)
		# 檢查是否為 ID=0 模式（收資料場景），或明確指定了負載顆數
		has_id0 = "0" in commands
		if load_count is not None:
			# 明確指定了負載顆數
			ts_line = f"{ts_line},{load_count}"
		elif has_id0:
			# ID=0 模式（收資料場景），預設加上負載顆數 0
			ts_line = f"{ts_line},0"
		
		# 如果有情況碼，加在第一行（時間戳之前）
		lines: List[str] = []
		if situation_code is not None:
			lines.append(str(int(situation_code)))
		lines.append(ts_line)
		
		# 2025/12/16 廠商更新：
		# - ID 不需要補到 10 顆，Command.txt 只需包含有指令的電池即可
		# - 若某顆電池沒有出現在 Command.txt 中，該顆電池的充電會自動關閉
		# 因此這裡僅輸出 commands 中出現的 PP
		# 格式：PP,功率(mW),流速,（3 個欄位）
		for pp in sorted(commands.keys()):
			# 有命令的電池：使用實際命令
			_, power_w, flow_percent = commands[pp]
			power_mw = int(round(float(power_w) * 1000.0))
			flow_int = int(round(max(0.0, min(100.0, float(flow_percent)))))
			
			# 格式：PP,功率(mW),流速,
			# 根據廠商確認：
			# - 功率：不補0（例如 8000 = 8000 mW）
			# - 流速：不補0，1~3位數都可以（例如 4 = 4%, 25 = 25%, 100 = 100%）
			power_str = f"{power_mw}"       # 不補0（例如 8000 -> 8000）
			flow_str = f"{flow_int}"        # 不補0，1~3位數（例如 4 -> 4, 25 -> 25, 100 -> 100）
			line = f"{pp},{power_str},{flow_str},"
			lines.append(line)
		
		content = "\n".join(lines) + "\n"
	
	# 原子寫入：先寫到臨時檔，再移動（避免寫入過程中被讀取）
	try:
		# 建立臨時檔
		dir_name = os.path.dirname(path) or "."
		base_name = os.path.basename(path)
		fd, temp_path = tempfile.mkstemp(
			prefix=base_name + ".",
			suffix=".tmp",
			dir=dir_name,
			text=True
		)
		
		try:
			# 寫入臨時檔（確保立即關閉）
			with os.fdopen(fd, "w", encoding="utf-8") as f:
				f.write(content)
				f.flush()  # 強制刷新緩衝區
				if hasattr(os, 'fsync'):
					try:
						os.fsync(f.fileno())  # 強制同步到磁碟（Unix/Linux）
					except (OSError, AttributeError):
						pass  # Windows 可能不支援 fsync
			# with 語句結束後，檔案會自動關閉
			
			# 在 Windows 上，確保檔案完全關閉後再移動（小延遲）
			time.sleep(0.01)  # 10ms 延遲，確保檔案系統釋放鎖定
			
			# 原子移動（Windows 上需要先刪除目標檔）
			if os.path.exists(path):
				try:
					os.remove(path)
				except (IOError, OSError):
					# 如果刪除失敗，嘗試覆寫
					pass
			
			shutil.move(temp_path, path)
			return True
			
		except Exception:
			# 清理臨時檔
			try:
				if os.path.exists(temp_path):
					os.remove(temp_path)
			except Exception:
				pass
			return False
			
	except Exception:
		# 如果原子寫入失敗，嘗試直接寫入（較不安全但作為備選）
		# 注意：直接寫入可能與廠商程式衝突，但作為最後手段
		try:
			with io.open(path, "w", encoding="utf-8") as f:
				f.write(content)
				f.flush()  # 強制刷新緩衝區
				if hasattr(os, 'fsync'):
					try:
						os.fsync(f.fileno())  # 強制同步到磁碟
					except (OSError, AttributeError):
						pass
			# 確保檔案關閉後再返回
			time.sleep(0.01)  # 10ms 延遲，確保檔案系統釋放鎖定
			return True
		except Exception:
			return False


# ---- 元件 → 模型（狀態）----
# 行格式：ID,YYYYMMDDhhmmss,VMMMMMMM,IMMMMMMM,FMMMMMMM,SMMMMMMM,RES1,RES2,RES3
# ID ∈ {PV, LD, B01..B99}

def parse_status_line(line: str) -> Tuple[str, datetime, float, float, float, float]:
	parts = [p.strip() for p in line.strip().split(",")]
	if len(parts) < 6:
		raise ValueError(f"Invalid status line: {line!r}")
	idv = parts[0]
	ts = parse_ts(parts[1])
	v = parse_unsigned_field(parts[2])     # V
	i = parse_signed_field(parts[3])       # A（0=正放電、1=負充電）
	f = parse_unsigned_field(parts[4]) if parts[4] else 0.0  # cc/min（只在電池適用）
	soc = parse_unsigned_field(parts[5]) if parts[5] else 0.0  # %
	return idv, ts, v, i, f, soc


def format_status_line(idv: str, ts: datetime, volt_v: float, curr_a: float, flow_ccm: float, soc_pct: float) -> str:
	return ",".join([
		idv,
		format_ts(ts),
		format_unsigned_field(volt_v),
		format_signed_field(curr_a),
		format_unsigned_field(flow_ccm),
		format_unsigned_field(soc_pct),
		" ", " ", " "
	])


def read_status_file(path: str, max_age_sec: Optional[int] = None) -> Dict[str, Tuple[datetime, float, float, float, float]]:
	"""
	讀取狀態文件（舊格式：每行含時間戳），回傳每個 ID 的最新一筆：
	{ ID: (ts, volt_v, curr_a, flow_ccm, soc_pct) }
	讀完清空檔案。
	"""
	results: Dict[str, Tuple[datetime, float, float, float, float]] = {}
	try:
		if not os.path.exists(path):
			return results
		with io.open(path, "r", encoding="utf-8") as f:
			lines = [ln for ln in f.readlines() if ln.strip()]
	except Exception:
		return results
	finally:
		# 清空檔案避免重複讀取
		try:
			with io.open(path, "w", encoding="utf-8") as w:
				w.write("")
		except Exception:
			pass

	now_ts = datetime.now(TZ_UTC8)
	for ln in lines:
		try:
			idv, ts, v, i, flow, soc = parse_status_line(ln)
		except Exception:
			continue
		if max_age_sec is not None and (now_ts - ts).total_seconds() > float(max_age_sec):
			continue
		if idv not in results or ts > results[idv][0]:
			results[idv] = (ts, v, i, flow, soc)
	return results


# ---- 廠商 Data File 格式（元件 → 模型）----
# 格式（2026/03 新版 — 含 MPPT-Bus + 負載）：
#   第一行：YYYYMMDDHHmmSS（時間戳）
#   第二行：SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,BusV,BusI,BusP,
#   第三行：LoadV,LoadI,LoadP,
#   第四行開始：ID,SOC,BV,BI,Temp,Speed,（電池資料）
#
# 向下相容舊格式（2025/12）：
#   第一行：YYYYMMDDHHmmSS
#   第二行：SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,
#   第三行開始：ID,SOC,BV,BI,Temp,Speed,
# 範例：01,101,500,1200,450,100,


class _VendorDataResult(dict):
	"""
	read_vendor_data_file 回傳型別。
	
	同時支援：
		1. 新版 dict 存取：result['mppt'], result['mppt_bus'], result['load'], result['batteries']
		2. 舊版 tuple 解構：mppt_data, battery_data = result  （向下相容）
	"""
	def __init__(self, mppt, mppt_bus, load, batteries, timestamp):
		super().__init__(
			mppt=mppt,
			mppt_bus=mppt_bus,
			load=load,
			batteries=batteries,
			timestamp=timestamp,
		)
		self._tuple = (mppt, batteries)  # 向下相容
	
	def __iter__(self):
		"""允許 mppt_data, batt_data = result"""
		return iter(self._tuple)
	
	def __len__(self):
		return 2  # 向下相容 tuple 長度

def parse_vendor_data_line(line: str) -> Tuple[str, float, float, float, float, float]:
	"""
	解析廠商 Data File 的一行（電池資料）。
	
	格式：PP, SOC, BV, BI, TEMP, Speed（6 欄位）
	
	單位（根據廠商確認，2025/12/28 更新）：
		- SOC: 0.1% (101 = 10.1%)
		- BV: 0.01V (500 = 5.00V)
		- BI: 1 mA，直接記錄原始數值（1200 = 1200 mA，不轉換為 A）
		- TEMP: 0.1°C (450 = 45.0°C)
		- Speed: 流速 0.1% (1000 = 100.0%)
	
	Returns:
		(pp, soc_pct, volt_v, curr_ma, temp_c, speed)
		curr_ma: 電池電流（毫安，原始數值）
	"""
	parts = [p.strip() for p in line.strip().split(",")]
	if len(parts) < 6:
		raise ValueError(f"Invalid vendor data line: {line!r}")
	# 確保只有6欄位（忽略多餘欄位）
	if len(parts) > 6:
		parts = parts[:6]
	
	pp = parts[0]
	soc_raw = int(parts[1]) if parts[1].isdigit() else 0
	soc_pct = float(soc_raw) / 10.0  # 0.1% 單位（101 = 10.1%）
	
	bv_raw = int(parts[2]) if parts[2].isdigit() else 0
	volt_v = float(bv_raw) / 100.0  # 0.01V 單位（500 = 5.00V）
	
	bi_raw = int(parts[3]) if parts[3].isdigit() else 0
	curr_ma = float(bi_raw)  # 1 mA 單位，直接記錄原始數值（1200 = 1200 mA）
	
	temp_raw = int(parts[4]) if parts[4].isdigit() else 0
	temp_c = float(temp_raw) / 10.0  # 0.1°C 單位（450 = 45.0°C）
	
	speed_raw = int(parts[5]) if parts[5].isdigit() else 0
	speed = float(speed_raw) / 10.0  # 流速 0.1% 單位（1000 = 100.0%）
	
	return pp, soc_pct, volt_v, curr_ma, temp_c, speed


def parse_mppt_line(line: str) -> Tuple[float, float, float, float, float, float]:
	"""
	解析 MPPT 行（太陽能板資料）— 舊版 6 欄位相容。
	
	格式：SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,
	
	單位（根據廠商確認，2025/12/28 更新）：
		- SolarV/MPPT_V: 0.01V（1600 = 16.00V）
		- SolarI/MPPT_I: 1mA，直接記錄原始數值（500 = 500 mA，不轉換為 A）
		- SolarP/MPPT_P: 1mW，直接記錄原始數值（8000 = 8000 mW，不轉換為 W）
	
	Returns:
		(solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw)
		solar_i_ma, mppt_i_ma: 電流（毫安，原始數值）
		solar_p_mw, mppt_p_mw: 功率（毫瓦，原始數值）
	"""
	parts = [p.strip() for p in line.strip().split(",") if p.strip()]
	if len(parts) < 6:
		raise ValueError(f"Invalid MPPT line: {line!r}")
	
	solar_v_raw = int(parts[0]) if parts[0].isdigit() else 0
	solar_v = float(solar_v_raw) / 100.0  # 0.01V 單位（1600 = 16.00V）
	
	solar_i_raw = int(parts[1]) if parts[1].isdigit() else 0
	solar_i_ma = float(solar_i_raw)  # 1mA 單位，直接記錄原始數值（500 = 500 mA）
	
	solar_p_raw = int(parts[2]) if parts[2].isdigit() else 0
	solar_p_mw = float(solar_p_raw)  # 1mW 單位，直接記錄原始數值（8000 = 8000 mW）
	
	mppt_v_raw = int(parts[3]) if parts[3].isdigit() else 0
	mppt_v = float(mppt_v_raw) / 100.0  # 0.01V 單位（1500 = 15.00V）
	
	mppt_i_raw = int(parts[4]) if parts[4].isdigit() else 0
	mppt_i_ma = float(mppt_i_raw)  # 1mA 單位，直接記錄原始數值（450 = 450 mA）
	
	mppt_p_raw = int(parts[5]) if parts[5].isdigit() else 0
	mppt_p_mw = float(mppt_p_raw)  # 1mW 單位，直接記錄原始數值（6750 = 6750 mW）
	
	return solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw


def parse_mppt_line_v2(line: str) -> Tuple[
	Tuple[float, float, float, float, float, float],
	Optional[Tuple[float, float, float]]
]:
	"""
	解析 MPPT 行（太陽能板資料）— 新版支援 MPPT-Bus（9 欄位）。
	
	格式（新版 2026/03）：
		SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,BusV,BusI,BusP,
	
	格式（舊版）：
		SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,
	
	單位（全部相同）：
		- V: 0.01V（1600 = 16.00V）
		- I: 1mA（500 = 500 mA）
		- P: 1mW（8000 = 8000 mW）
	
	Returns:
		(mppt_6tuple, mppt_bus_3tuple_or_None)
		mppt_6tuple: (solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw)
		mppt_bus:    (bus_v, bus_i_ma, bus_p_mw) 或 None（舊格式）
	"""
	parts = [p.strip() for p in line.strip().split(",") if p.strip()]
	if len(parts) < 6:
		raise ValueError(f"Invalid MPPT line: {line!r}")
	
	# 前 6 欄：Solar + MPPT
	solar_v = float(int(parts[0]) if parts[0].isdigit() else 0) / 100.0
	solar_i_ma = float(int(parts[1]) if parts[1].isdigit() else 0)
	solar_p_mw = float(int(parts[2]) if parts[2].isdigit() else 0)
	mppt_v = float(int(parts[3]) if parts[3].isdigit() else 0) / 100.0
	mppt_i_ma = float(int(parts[4]) if parts[4].isdigit() else 0)
	mppt_p_mw = float(int(parts[5]) if parts[5].isdigit() else 0)
	
	mppt_6 = (solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw)
	
	# 後 3 欄：MPPT-Bus（新版格式 ≥9 欄位）
	mppt_bus = None
	if len(parts) >= 9:
		bus_v = float(int(parts[6]) if parts[6].isdigit() else 0) / 100.0
		bus_i_ma = float(int(parts[7]) if parts[7].isdigit() else 0)
		bus_p_mw = float(int(parts[8]) if parts[8].isdigit() else 0)
		mppt_bus = (bus_v, bus_i_ma, bus_p_mw)
	
	return mppt_6, mppt_bus


def parse_load_line(line: str) -> Tuple[float, float, float]:
	"""
	解析負載功耗行（2026/03 新增）。
	
	格式：LoadV,LoadI,LoadP,
	
	單位（與 MPPT 相同）：
		- V: 0.01V（1200 = 12.00V）
		- I: 1mA（5500 = 5500 mA）
		- P: 1mW（6600 = 6600 mW）
	
	Returns:
		(load_v, load_i_ma, load_p_mw)
	"""
	parts = [p.strip() for p in line.strip().split(",") if p.strip()]
	if len(parts) < 3:
		raise ValueError(f"Invalid load line: {line!r}")
	
	load_v = float(int(parts[0]) if parts[0].isdigit() else 0) / 100.0   # 0.01V
	load_i_ma = float(int(parts[1]) if parts[1].isdigit() else 0)        # mA
	load_p_mw = float(int(parts[2]) if parts[2].isdigit() else 0)        # mW
	
	return load_v, load_i_ma, load_p_mw


def read_vendor_data_file(path: str, max_age_sec: Optional[int] = None, 
                          clear_after_read: bool = True) -> Dict:
	"""
	讀取廠商 Data File（Data.txt），回傳所有解析資料。
	
	格式（2026/03 新版 — 向下相容舊版）：
		第一行：YYYYMMDDHHmmSS（時間戳）
		第二行：SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P[,BusV,BusI,BusP],
		第三行：LoadV,LoadI,LoadP,                      ← 新增（若有 MPPT-Bus）
		之後：  ID,SOC,BV,BI,Temp,Speed,（電池資料）
	
	舊格式（2025/12）：
		第一行：YYYYMMDDHHmmSS
		第二行：SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,
		第三行開始：ID,SOC,BV,BI,Temp,Speed,
	
	重要：廠商程式會每秒寫入檔案，讀取時需注意同步問題。
	
	Args:
		path: 檔案路徑（應為 Data.txt）
		max_age_sec: 最大年齡（秒），使用第一行時間戳判斷
		clear_after_read: 讀取後是否清空檔案（避免重複讀取）
	
	Returns:
		dict with keys:
			'mppt':      (solar_v, solar_i_ma, solar_p_mw, mppt_v, mppt_i_ma, mppt_p_mw) or None
			'mppt_bus':  (bus_v, bus_i_ma, bus_p_mw) or None  — 新欄位
			'load':      (load_v, load_i_ma, load_p_mw) or None  — 新欄位
			'batteries': { PP: (ts, soc_pct, volt_v, curr_ma, temp_c, speed) }
			'timestamp': datetime or None
		
		向下相容：亦可用 tuple 解構 → mppt_data, battery_data = result
		（會忽略 mppt_bus / load，不影響舊程式碼）
	"""
	results: Dict[str, Tuple[datetime, float, float, float, float, float]] = {}
	mppt_data: Optional[Tuple[float, float, float, float, float, float]] = None
	mppt_bus_data: Optional[Tuple[float, float, float]] = None
	load_data: Optional[Tuple[float, float, float]] = None
	file_ts: Optional[datetime] = None
	read_ts = datetime.now(TZ_UTC8)
	
	try:
		if not os.path.exists(path):
			return _VendorDataResult(mppt_data, mppt_bus_data, load_data, results, file_ts)
		
		lines: List[str] = []
		try:
			with io.open(path, "r", encoding="utf-8") as f:
				lines = [ln for ln in f.readlines() if ln.strip()]
		except (IOError, OSError, PermissionError):
			return _VendorDataResult(mppt_data, mppt_bus_data, load_data, results, file_ts)
		
		if not lines:
			return _VendorDataResult(mppt_data, mppt_bus_data, load_data, results, file_ts)
		
		# ── 第一行：時間戳 ──
		file_ts = read_ts
		try:
			first_line = lines[0].strip()
			if len(first_line) >= 14:
				ts_part = first_line[:14]
				if ts_part.isdigit():
					file_ts = parse_ts(ts_part)
					lines = lines[1:]
		except Exception:
			pass
		
		# 檢查時間戳是否過期
		if max_age_sec is not None:
			age_sec = (datetime.now(TZ_UTC8) - file_ts).total_seconds()
			if age_sec > float(max_age_sec):
				return _VendorDataResult(None, None, None, {}, file_ts)
		
		# ── 第二行：MPPT（可能含 MPPT-Bus → 9 欄位） ──
		if lines:
			first_data_line = lines[0].strip()
			first_parts = [p.strip() for p in first_data_line.split(",") if p.strip()]
			if len(first_parts) >= 6:
				first_field = first_parts[0]
				if not (first_field.isdigit() and 1 <= int(first_field) <= 10):
					try:
						mppt_6, mppt_bus_data = parse_mppt_line_v2(first_data_line)
						mppt_data = mppt_6
						lines = lines[1:]
					except Exception:
						lines = lines[1:]
		
		# ── 第三行（新格式）：負載功耗（3 欄位，不以電池 ID 開頭） ──
		if lines and mppt_bus_data is not None:
			# 只有在新格式（有 MPPT-Bus）時才嘗試解析負載行
			load_line = lines[0].strip()
			load_parts = [p.strip() for p in load_line.split(",") if p.strip()]
			if len(load_parts) >= 3:
				first_field = load_parts[0]
				# 負載行不是電池 ID（1-10 開頭且有 6 欄位）
				is_battery = (first_field.isdigit() and 1 <= int(first_field) <= 10 
				              and len(load_parts) >= 6)
				if not is_battery:
					try:
						load_data = parse_load_line(load_line)
						lines = lines[1:]
					except Exception:
						lines = lines[1:]
		
		# ── 之後：電池資料 ──
		for ln in lines:
			try:
				pp, soc_pct, volt_v, curr_a, temp_c, speed = parse_vendor_data_line(ln)
			except Exception:
				continue
			results[pp] = (file_ts, soc_pct, volt_v, curr_a, temp_c, speed)
		
	except Exception:
		return _VendorDataResult(mppt_data, mppt_bus_data, load_data, results, file_ts)
	finally:
		if clear_after_read:
			try:
				with io.open(path, "w", encoding="utf-8") as w:
					w.write("")
					w.flush()
					if hasattr(os, 'fsync'):
						try:
							os.fsync(w.fileno())
						except (OSError, AttributeError):
							pass
				time.sleep(0.01)
			except (IOError, OSError, PermissionError):
				pass
	
	return _VendorDataResult(mppt_data, mppt_bus_data, load_data, results, file_ts)


