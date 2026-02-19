# -*- coding: utf-8 -*-
"""TTS Pro V2.3.5"""

from __future__ import annotations

import atexit
import asyncio
import base64
import contextlib
import gc
import json
import logging
import multiprocessing
import os
import queue
import random
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# ═══════════════════════════════════════════════════════════════
#  IMPORTS — curl_cffi primary, aiohttp fallback
# ═══════════════════════════════════════════════════════════════

CURL_OK = False
try:
    from curl_cffi.requests import AsyncSession as CurlSession
    CURL_OK = True
except ImportError:
    CurlSession = None

AIOHTTP_OK = False
try:
    import aiohttp
    AIOHTTP_OK = True
except ImportError:
    aiohttp = None

WEBVIEW_OK = False
try:
    import webview
    WEBVIEW_OK = True
except Exception:
    webview = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import orjson
    _jl = lambda b: orjson.loads(b if isinstance(b, (bytes, bytearray)) else b.encode())
    _jd = lambda o: orjson.dumps(o).decode()
except ImportError:
    _jl = lambda b: json.loads(b.decode() if isinstance(b, (bytes, bytearray)) else b)
    _jd = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))

if sys.platform != "win32":
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

# ═══════════════════════════════════════════════════════════════
#  PROCESS SAFETY
# ═══════════════════════════════════════════════════════════════

_FF_PIDS: Set[int] = set()
_FF_LOCK = threading.Lock()
CURRENT_UI = None


def _reg_pid(pid: int):
    with _FF_LOCK: _FF_PIDS.add(pid)

def _unreg_pid(pid: int):
    with _FF_LOCK: _FF_PIDS.discard(pid)

def _kill_all():
    with _FF_LOCK: pids = list(_FF_PIDS)
    for p in pids:
        try: os.kill(p, signal.SIGTERM)
        except Exception: pass

atexit.register(_kill_all)
try:
    signal.signal(signal.SIGTERM, lambda s, f: (_kill_all(), sys.exit(0)))
    if sys.platform != "win32":
        signal.signal(signal.SIGINT, lambda s, f: (_kill_all(), sys.exit(0)))
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════
#  HARDWARE PROFILER — Clean
# ═══════════════════════════════════════════════════════════════

@dataclass(slots=True)
class HW:
    cores: int = 2
    ram_gb: float = 4.0
    tier: str = "STANDARD"
    rps: int = 50
    concurrency: int = 40
    conn_pool: int = 80
    merge_batch: int = 60
    merge_workers: int = 3
    ff_threads: int = 2
    merge_trigger: int = 35

    @staticmethod
    def detect() -> "HW":
        h = HW()
        h.cores = os.cpu_count() or 2
        if psutil:
            try: h.ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
            except Exception: pass
        else:
            try:
                if sys.platform == "win32":
                    import ctypes
                    class MS(ctypes.Structure):
                        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong),
                                    ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                                    ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                                    ("ex", ctypes.c_ulonglong)]
                    s = MS(l=ctypes.sizeof(MS))
                    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
                    h.ram_gb = round(s.total / (1024**3), 1)
                else:
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemTotal"):
                                h.ram_gb = round(int(line.split()[1]) / (1024**2), 1)
                                break
            except Exception: pass

        r, c = h.ram_gb, h.cores
        if r >= 16 and c >= 8:    h.tier = "ULTRA"
        elif r >= 12 and c >= 6:  h.tier = "HIGH"
        elif r >= 6 and c >= 4:   h.tier = "STANDARD"
        else:                     h.tier = "LOW"

        T = {
            "LOW":      (35, 30, 50, 40, 2, 1, 25),
            "STANDARD": (55, 40, 80, 70, 3, 2, 35),
            "HIGH":     (65, 40, 100, 100, 4, 3, 45),
            "ULTRA":    (72, 40, 120, 130, 6, 4, 55),
        }
        v = T[h.tier]
        h.rps, h.concurrency, h.conn_pool, h.merge_batch, h.merge_workers, h.ff_threads, h.merge_trigger = v
        return h

    @property
    def transport(self) -> str:
        return "curl_cffi/H2" if CURL_OK else "aiohttp/H1.1"

    def summary(self) -> str:
        return f"{self.tier} | {self.cores}C | {self.ram_gb}GB | {self.transport} | RPS:{self.rps}"


HW_PROFILE = HW.detect()

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

@dataclass(slots=True, frozen=True)
class CFG:
    APP_NAME: str = "TTS PRO"
    VERSION: str = "3.0"
    ROOT_DIR: Path = field(default_factory=lambda: Path(os.environ.get("DUB_PRO_ROOT", "DATA")))

    API_URL: str = "https://api.vivibe.app/json-rpc"
    REFERER: str = "https://www.vivibe.app/"
    DEFAULT_VOICE: str = "jpXdqzDzqAVcV1fLRoECAV"
    SPEED: float = 1.3

    RETRY: int = 4
    RETRY_BASE: float = 0.05
    TIMEOUT: float = 15.0
    TIMEOUT_CONN: float = 3.0

    CB_FAIL: int = 6
    CB_HALF_OPEN: float = 1.5
    HEDGE_AFTER: float = 3.0

    CURL_IMPERSONATE: str = "chrome136"

    FL_KEY: str = "AIzaSyBEfuL7qePYp9WlBFPjVLXLKN5Us6rr6tg"
    FL_EMAIL: str = "p.h.a.n.k.i.e.t.1.2.3.3@gmail.com"
    FL_PASS: str = "p.h.a.n.k.i.e.t.1.2.3.3@gmail.com"

    @property
    def FL_URL(self) -> str:
        return f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.FL_KEY}"

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/136.0.0.0 Safari/537.36",
            "Origin": "https://www.vivibe.app",
            "Referer": self.REFERER,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Ch-Ua": '"Chromium";v="136","Google Chrome";v="136","Not.A/Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
        }


C = CFG()
C.ROOT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  LOGGER — Minimal
# ═══════════════════════════════════════════════════════════════

_log = logging.getLogger(C.APP_NAME)
_log.setLevel(logging.INFO)
if not _log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _log.addHandler(_h)

# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

def _fmt(sec: int) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _rmtree(path: Union[str, Path]):
    p = Path(path)
    if not p.exists(): return
    for i in range(3):
        try: shutil.rmtree(p); return
        except Exception: time.sleep(0.05 * (i + 1))
    shutil.rmtree(p, ignore_errors=True)


def _srt_ms(t: str) -> int:
    t = t.strip().replace(",", ".")
    parts = t.split(":")
    if len(parts) != 3: return 0
    try:
        h, m, sec = parts
        sp = sec.split(".", 1)
        s = int(sp[0])
        ms = int(sp[1][:3].ljust(3, "0")) if len(sp) > 1 else 0
        return (int(h) * 3600 + int(m) * 60 + s) * 1000 + ms
    except Exception: return 0


def parse_srt(path: Path) -> List[dict]:
    try:
        import pysrt
        subs = pysrt.open(str(path), encoding="utf-8")
        items = []
        for i, sub in enumerate(subs, 1):
            txt = " ".join(str(sub.text or "").splitlines()).strip()
            if txt:
                items.append({"id": i, "ms": int(sub.start.ordinal), "text": txt})
        if items: return items
    except Exception: pass

    content = path.read_text("utf-8-sig", errors="ignore").replace("\r\n", "\n")
    lines = content.split("\n")
    items, i, n = [], 0, len(lines)
    while i < n:
        while i < n and not lines[i].strip(): i += 1
        if i >= n: break
        idx = lines[i].strip(); i += 1
        if not idx.isdigit(): continue
        if i >= n: break
        tl = lines[i].strip(); i += 1
        if "-->" not in tl: continue
        ms = _srt_ms(tl.split("-->")[0])
        buf = []
        while i < n and lines[i].strip():
            buf.append(lines[i].strip()); i += 1
        txt = " ".join(buf).strip()
        if txt: items.append({"id": int(idx), "ms": ms, "text": txt})
    return items


# ═══════════════════════════════════════════════════════════════
#  WAV UTILS
# ═══════════════════════════════════════════════════════════════

class Wav:
    @staticmethod
    def header(data: bytes) -> Optional[dict]:
        if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE": return None
        try:
            pos, fmt, doff, dsz = 12, None, 0, 0
            while pos < len(data) - 8:
                cid = data[pos:pos+4]
                csz = struct.unpack_from("<I", data, pos+4)[0]
                if cid == b"fmt " and csz >= 16:
                    af, ch, sr, br, ba, bits = struct.unpack_from("<HHIIHH", data, pos+8)
                    fmt = {"channels": ch, "sample_rate": sr, "byte_rate": br,
                           "block_align": ba, "bits": bits}
                elif cid == b"data":
                    doff, dsz = pos+8, csz; break
                pos += 8 + csz + (csz % 2)
            if fmt and doff:
                fmt["data_offset"], fmt["data_size"] = doff, dsz
                return fmt
        except Exception: pass
        return None

    @staticmethod
    def pcm(wav: bytes) -> Optional[Tuple[bytes, int, int, int]]:
        h = Wav.header(wav)
        if not h: return None
        o, s = h["data_offset"], min(h["data_size"], len(wav) - h["data_offset"])
        return wav[o:o+s], h["sample_rate"], h["channels"], h["bits"]

    @staticmethod
    def make_header(sz: int, sr: int = 44100, ch: int = 1, bits: int = 16) -> bytes:
        return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36+sz, b"WAVE", b"fmt ", 16,
                           1, ch, sr, sr*ch*bits//8, ch*bits//8, bits, b"data", sz)


# ═══════════════════════════════════════════════════════════════
#  AUTH — Lazy, one-shot, auto-refresh when needed
# ═══════════════════════════════════════════════════════════════

class Auth:
    _TOKEN_PATH = Path.home() / ".tts_pro" / "session.tok"
    _lock = threading.Lock()
    _token: Optional[str] = None

    @classmethod
    def get(cls) -> str:
        with cls._lock:
            if cls._valid(cls._token): return cls._token
            disk = cls._load()
            if cls._valid(disk):
                cls._token = disk
                return disk
            new = cls._login()
            if new:
                cls._token = new
                cls._save(new)
                return new
            return ""

    @classmethod
    def refresh(cls) -> str:
        with cls._lock:
            new = cls._login()
            if new:
                cls._token = new
                cls._save(new)
                return new
            return ""

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._token = None
            try: cls._TOKEN_PATH.unlink(missing_ok=True)
            except Exception: pass

    @classmethod
    def _valid(cls, tok: Optional[str]) -> bool:
        if not tok: return False
        t = tok.split(" ", 1)[-1].strip() if tok.lower().startswith("bearer ") else tok.strip()
        parts = t.split(".")
        if len(parts) != 3: return bool(t)
        try:
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + "==").decode("utf-8", "ignore"))
            exp = payload.get("exp")
            if exp is not None: return (float(exp) - time.time()) > 300
            return True
        except Exception: return bool(t)

    @classmethod
    def _login(cls) -> Optional[str]:
        for i in range(3):
            try:
                data = _jd({"email": C.FL_EMAIL, "password": C.FL_PASS,
                            "returnSecureToken": True, "clientType": "CLIENT_TYPE_WEB"}).encode()
                req = urllib.request.Request(C.FL_URL, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    if r.status == 200: return _jl(r.read()).get("idToken")
            except Exception:
                if i < 2: time.sleep(0.5 * (i + 1))
        return None

    @classmethod
    def _load(cls) -> str:
        try: return cls._TOKEN_PATH.read_text("utf-8").strip() if cls._TOKEN_PATH.exists() else ""
        except Exception: return ""

    @classmethod
    def _save(cls, tok: str):
        try:
            cls._TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = cls._TOKEN_PATH.with_suffix(".tmp")
            tmp.write_text(tok, "utf-8")
            os.replace(tmp, cls._TOKEN_PATH)
        except Exception: pass


# ═══════════════════════════════════════════════════════════════
#  SETTINGS
# ═══════════════════════════════════════════════════════════════

def _default_out() -> Path:
    if sys.platform == "win32":
        for d in ("D:/", os.environ.get("SystemDrive", "C:") + "/"):
            if Path(d).exists(): return Path(d)
    return Path.home()


@dataclass
class State:
    output_dir: Path = field(default_factory=_default_out)
    voice_id: str = field(default_factory=lambda: C.DEFAULT_VOICE)


class Settings:
    _P = C.ROOT_DIR / "__settings__.json"

    @staticmethod
    def load() -> State:
        s = State()
        try:
            d = _jl(Settings._P.read_bytes())
            if d.get("output_dir"): s.output_dir = Path(d["output_dir"])
            if d.get("voice_id"): s.voice_id = str(d["voice_id"]).strip()
        except Exception: pass
        return s

    @staticmethod
    def save(s: State):
        try:
            tmp = Settings._P.with_suffix(".tmp")
            tmp.write_text(_jd({"output_dir": str(s.output_dir), "voice_id": s.voice_id}), "utf-8")
            os.replace(tmp, Settings._P)
        except Exception: pass


# ═══════════════════════════════════════════════════════════════
#  FFMPEG
# ═══════════════════════════════════════════════════════════════

class FF:
    _norm: Optional[bool] = None
    _samples: Optional[bool] = None

    @staticmethod
    def bins() -> Tuple[str, Optional[str]]:
        def _l(n):
            if sys.platform == "win32":
                p = Path(os.getcwd()) / f"{n}.exe"
                return str(p) if p.exists() else None
            return None
        ff = _l("ffmpeg") or shutil.which("ffmpeg")
        if not ff: raise FileNotFoundError("FFmpeg not found")
        return ff, _l("ffprobe") or shutil.which("ffprobe")

    @staticmethod
    def _si() -> dict:
        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            return {"startupinfo": si, "creationflags": subprocess.CREATE_NO_WINDOW}
        return {}

    @staticmethod
    def has_norm(ff: str) -> bool:
        if FF._norm is None:
            try:
                out = subprocess.check_output([ff, "-hide_banner", "-h", "filter=amix"],
                                              stderr=subprocess.STDOUT, text=True, timeout=10, **FF._si())
                FF._norm = "normalize" in out
            except Exception: FF._norm = False
        return FF._norm

    @staticmethod
    def has_samples(ff: str) -> bool:
        if FF._samples is None:
            try:
                out = subprocess.check_output([ff, "-hide_banner", "-h", "filter=adelay"],
                                              stderr=subprocess.STDOUT, text=True, timeout=10, **FF._si())
                FF._samples = "append 's'" in out.lower() and "samples" in out.lower()
            except Exception: FF._samples = False
        return FF._samples

    @staticmethod
    def sr(path: str, probe: Optional[str]) -> int:
        if not probe: return 44100
        try:
            raw = subprocess.check_output(
                [probe, "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", "a:0", path],
                timeout=10, **FF._si())
            return int(_jl(raw)["streams"][0].get("sample_rate", 44100))
        except Exception: return 44100

    @staticmethod
    def merge_cmd(ff: str, inputs: List[str], delays: List[int], out: str,
                  sr: int, use_s: bool, norm: bool, threads: int = 2) -> List[str]:
        n = len(inputs)
        cmd = [ff, "-y", "-hide_banner", "-nostdin", "-v", "error", "-threads", str(threads)]
        for i in inputs: cmd.extend(["-i", i])

        if n == 1:
            d = delays[0]
            if d > 0:
                dv = f"{(d*sr)//1000}s|{(d*sr)//1000}s" if use_s else f"{d}|{d}"
                cmd.extend(["-filter_complex", f"[0:a]adelay={dv}[out]", "-map", "[out]"])
            else:
                cmd.extend(["-map", "0:a"])
            cmd.extend(["-c:a", "pcm_s16le", "-ar", str(sr), "-ac", "1", out])
            return cmd

        filters, labels = [], []
        for i, d in enumerate(delays):
            dv = f"{(d*sr)//1000}s|{(d*sr)//1000}s" if use_s else f"{d}|{d}"
            filters.append(f"[{i}:a]adelay={dv}[a{i}]")
            labels.append(f"[a{i}]")
        amix = f"inputs={n}:duration=longest:dropout_transition=0"
        if norm: amix += ":normalize=0"
        filters.append(f"{''.join(labels)}amix={amix}[out]")
        cmd.extend(["-filter_complex", ";".join(filters), "-map", "[out]",
                    "-c:a", "pcm_s16le", "-ar", str(sr), "-ac", "1", out])
        return cmd


# ═══════════════════════════════════════════════════════════════
#  RATE CONTROLLER
# ═══════════════════════════════════════════════════════════════

class RateCtrl:
    def __init__(self, initial: int, cap: int = 75):
        self._target = min(initial, cap)
        self._cap = cap
        self._rps = self._target
        self._tokens = float(self._target)
        self._max_tok = self._target * 2.0
        self._last = 0.0
        self._c429 = 0
        self._t429 = 0
        self._last429 = 0.0
        self._last_ramp = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self._lock:
                now = time.monotonic()
                if self._last > 0:
                    self._tokens = min(self._max_tok, self._tokens + (now - self._last) * self._rps)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            await asyncio.sleep(min(0.5, 1.0 / max(self._rps, 1)))

    async def on_429(self):
        async with self._lock:
            self._c429 += 1; self._t429 += 1; self._last429 = time.monotonic()
            new = max(8, int(self._rps * 0.55))
            if new != self._rps:
                self._rps = new
                self._max_tok = new * 2.0
                self._tokens = min(self._tokens, self._max_tok)

    async def on_ok(self):
        async with self._lock:
            self._c429 = 0
            now = time.monotonic()
            if ((now - self._last429 if self._last429 else 999) > 5.0 and
                (now - self._last_ramp if self._last_ramp else 999) > 2.5 and
                self._rps < self._target):
                self._rps = min(self._target, self._rps + max(2, int(self._rps * 0.12)))
                self._max_tok = self._rps * 2.0
                self._last_ramp = now

    @property
    def rps(self) -> int: return self._rps
    @property
    def total_429(self) -> int: return self._t429


# ═══════════════════════════════════════════════════════════════
#  CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════

class CB:
    def __init__(self, threshold: int = 6, half_open: float = 1.5):
        self._st = "CLOSED"
        self._fails = 0
        self._th = threshold
        self._ho = half_open
        self._last_fail = 0.0
        self._ok_streak = 0
        self._lock = asyncio.Lock()

    async def ok(self) -> bool:
        async with self._lock:
            if self._st == "CLOSED": return True
            if self._st == "OPEN" and time.monotonic() - self._last_fail > self._ho:
                self._st = "HALF_OPEN"; return True
            return self._st == "HALF_OPEN"

    async def success(self):
        async with self._lock:
            self._ok_streak += 1
            if self._st == "HALF_OPEN" and self._ok_streak >= 2:
                self._st = "CLOSED"; self._fails = 0; self._ok_streak = 0
            elif self._st == "CLOSED":
                self._fails = max(0, self._fails - 1)

    async def fail(self):
        async with self._lock:
            self._fails += 1; self._ok_streak = 0; self._last_fail = time.monotonic()
            if self._fails >= self._th: self._st = "OPEN"

    async def wait(self):
        while not await self.ok(): await asyncio.sleep(0.05)

    @property
    def state(self) -> str: return self._st


# ═══════════════════════════════════════════════════════════════
#  TTS PIPELINE
# ═══════════════════════════════════════════════════════════════

class Pipeline:
    def __init__(self):
        self.ff, self.probe = FF.bins()
        self.norm = FF.has_norm(self.ff)
        self.use_s = FF.has_samples(self.ff)
        self.session = None
        self._curl = CURL_OK
        self._sr = 44100; self._ch = 1; self._bits = 16

    async def __aenter__(self):
        if self._curl:
            self.session = CurlSession(impersonate=C.CURL_IMPERSONATE, verify=False,
                                       timeout=C.TIMEOUT, max_clients=HW_PROFILE.conn_pool)
        elif AIOHTTP_OK:
            conn = aiohttp.TCPConnector(limit=HW_PROFILE.conn_pool, limit_per_host=40,
                                        enable_cleanup_closed=True, ttl_dns_cache=7200, ssl=False)
            self.session = aiohttp.ClientSession(
                connector=conn, timeout=aiohttp.ClientTimeout(total=C.TIMEOUT, connect=C.TIMEOUT_CONN),
                json_serialize=_jd)
        else:
            raise RuntimeError("No HTTP lib! pip install curl-cffi OR aiohttp")
        return self

    async def __aexit__(self, *exc):
        if self.session:
            try: await self.session.close()
            except Exception: pass

    async def _post(self, url, data, hdrs) -> Optional[Tuple[int, Any]]:
        try:
            if self._curl:
                r = await self.session.post(url, json=data, headers=hdrs, timeout=C.TIMEOUT)
                return (r.status_code, r.json())
            else:
                async with self.session.post(url, json=data, headers=hdrs) as r:
                    if r.status in (200, 429): return (r.status, await r.json())
                    return (r.status, None)
        except asyncio.CancelledError: raise
        except Exception: return None

    async def _get(self, url) -> Optional[bytes]:
        try:
            if self._curl:
                r = await self.session.get(url, headers={}, timeout=C.TIMEOUT)
                return r.content if r.status_code == 200 else None
            else:
                async with self.session.get(url) as r:
                    return await r.read() if r.status == 200 else None
        except asyncio.CancelledError: raise
        except Exception: return None

    async def process(self, items: List[dict], out: Path, voice: str, token: str,
                      pq: Optional[queue.Queue] = None, stop: Optional[threading.Event] = None) -> Optional[str]:
        total = len(items)
        if not total: return None

        audio: Dict[int, bytes] = {}
        done: Set[int] = set()
        fails = 0
        lock = asyncio.Lock()
        t0 = time.monotonic()
        tmpdir = Path(tempfile.mkdtemp(prefix="tts_"))
        chunks: List[Tuple[int, Path]] = []
        merged: Set[int] = set()
        cidx = 0

        rate = RateCtrl(HW_PROFILE.rps)
        cb = CB(C.CB_FAIL, C.CB_HALF_OPEN)
        sem = asyncio.Semaphore(min(HW_PROFILE.concurrency, 40))
        msem = asyncio.Semaphore(HW_PROFILE.merge_workers)
        by_id = {it["id"]: it for it in items}
        sr_set = asyncio.Event()

        hdrs = {**C.headers, "Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        async def _call(text: str) -> Optional[Tuple[bytes, int]]:
            payload = {"method": "tts", "input": {"text": text, "userVoiceId": voice,
                                                   "speed": C.SPEED, "blockVersion": 0}}
            r = await self._post(C.API_URL, payload, hdrs)
            if not r: return None
            st, data = r
            if st == 429 or st != 200 or not data: return None
            url = data.get("result", {}).get("url") if isinstance(data, dict) else None
            if not url: return None
            raw = await self._get(url)
            return (raw, 200) if raw and len(raw) > 44 else None

        async def _one(item: dict):
            nonlocal fails
            iid, text = item["id"], item.get("text", "").strip()
            if not text or (stop and stop.is_set()): return

            async with sem:
                for att in range(C.RETRY):
                    if stop and stop.is_set(): return
                    async with lock:
                        if iid in done: return

                    await cb.wait()
                    await rate.acquire()

                    result = None
                    if att == 0:
                        primary = asyncio.create_task(_call(text))
                        try:
                            result = await asyncio.wait_for(asyncio.shield(primary), timeout=C.HEDGE_AFTER)
                        except asyncio.TimeoutError:
                            await rate.acquire()
                            hedge = asyncio.create_task(_call(text))
                            d_set, p_set = await asyncio.wait({primary, hedge}, return_when=asyncio.FIRST_COMPLETED)
                            for t in d_set:
                                try:
                                    r = t.result()
                                    if r and r[1] == 200: result = r; break
                                except Exception: pass
                            for t in p_set:
                                t.cancel()
                                with contextlib.suppress(asyncio.CancelledError): await t
                        except asyncio.CancelledError:
                            primary.cancel(); return
                    else:
                        result = await _call(text)

                    if not result:
                        await rate.on_429(); await cb.fail()
                        if att < C.RETRY - 1:
                            await asyncio.sleep(C.RETRY_BASE * (2**att) + random.random() * 0.03)
                        continue

                    raw, _ = result
                    await cb.success(); await rate.on_ok()

                    async with lock:
                        if iid in done: return
                        audio[iid] = raw; done.add(iid)
                        ld, lf = len(done), fails

                    if not sr_set.is_set():
                        h = Wav.header(raw)
                        if h:
                            self._sr = h.get("sample_rate", 44100)
                            self._ch = h.get("channels", 1)
                            self._bits = h.get("bits", 16)
                        sr_set.set()

                    if pq:
                        el = time.monotonic() - t0
                        ips = ld / el if el > 0 else 0
                        eta = (total - ld) / ips if ips > 0 else 0
                        pq.put(("PROG", (ld, total, lf, ips, rate.rps, cb.state, eta)))
                    return

                async with lock:
                    fails += 1

        async def _merge_batch(batch: List[dict], ci: int) -> Optional[Tuple[int, Path]]:
            async with msem:
                try:
                    batch = sorted(batch, key=lambda x: x["ms"])
                    if not batch: return None
                    base = batch[0]["ms"]
                    opath = tmpdir / f"c_{ci:05d}.wav"
                    bdir = tmpdir / f"_b{ci}"; bdir.mkdir(exist_ok=True)
                    inps, dls = [], []
                    for it in batch:
                        async with lock: d = audio.get(it["id"])
                        if not d: continue
                        fp = bdir / f"{it['id']}.wav"; fp.write_bytes(d)
                        inps.append(str(fp)); dls.append(it["ms"] - base)
                    if not inps: shutil.rmtree(bdir, ignore_errors=True); return None
                    cmd = FF.merge_cmd(self.ff, inps, dls, str(opath), self._sr, self.use_s, self.norm, HW_PROFILE.ff_threads)
                    proc = await asyncio.create_subprocess_exec(
                        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE, **FF._si())
                    _reg_pid(proc.pid)
                    await asyncio.wait_for(proc.communicate(), timeout=120)
                    _unreg_pid(proc.pid)
                    shutil.rmtree(bdir, ignore_errors=True)
                    if opath.exists() and opath.stat().st_size > 44:
                        async with lock:
                            for it in batch: audio.pop(it["id"], None)
                        return (base, opath)
                    return None
                except Exception: return None

        mtasks: List[asyncio.Task] = []

        async def _monitor():
            nonlocal cidx
            while True:
                if stop and stop.is_set(): break
                await asyncio.sleep(0.3)
                async with lock:
                    um = done - merged
                    fin = len(done) + fails >= total
                if len(um) < HW_PROFILE.merge_trigger and not fin: continue
                if not um:
                    if fin: break
                    continue
                ul = sorted([by_id[u] for u in um if u in by_id], key=lambda x: x["ms"])
                for i in range(0, len(ul), HW_PROFILE.merge_batch):
                    b = ul[i:i+HW_PROFILE.merge_batch]
                    if len(b) < 3 and not fin and len(ul) > HW_PROFILE.merge_batch: continue
                    bids = {it["id"] for it in b}
                    async with lock: merged.update(bids)
                    ci = cidx; cidx += 1
                    mtasks.append(asyncio.create_task(_merge_batch(b, ci)))
                if fin: break

        if pq:
            pq.put(("LOG", f"Pipeline: {total} items | {HW_PROFILE.transport} | Sem:{min(HW_PROFILE.concurrency,40)}", "sys"))

        tasks = [asyncio.create_task(_one(it)) for it in items]
        mon = asyncio.create_task(_monitor())
        await asyncio.gather(*tasks, return_exceptions=True)

        if pq:
            el = time.monotonic() - t0
            pq.put(("LOG", f"TTS complete: {len(done)}/{total} in {el:.1f}s | 429s: {rate.total_429}", "ok"))

        async with lock: fu = done - merged
        if fu:
            fl = sorted([by_id[u] for u in fu if u in by_id], key=lambda x: x["ms"])
            for i in range(0, len(fl), HW_PROFILE.merge_batch):
                b = fl[i:i+HW_PROFILE.merge_batch]
                ci = cidx; cidx += 1
                mtasks.append(asyncio.create_task(_merge_batch(b, ci)))
                async with lock: merged.update(it["id"] for it in b)

        mon.cancel()
        with contextlib.suppress(asyncio.CancelledError): await mon

        if mtasks:
            results = await asyncio.gather(*mtasks, return_exceptions=True)
            for r in results:
                if isinstance(r, tuple) and len(r) == 2: chunks.append(r)

        if not chunks and not audio:
            shutil.rmtree(tmpdir, ignore_errors=True); return None

        result = await self._stitch(chunks, audio, by_id, out, tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)
        audio.clear(); gc.collect()
        return result

    async def _stitch(self, chunks, audio, by_id, out, tmpdir) -> Optional[str]:
        sr = self._sr
        sc = sorted(chunks, key=lambda x: x[0])

        if len(sc) == 1:
            _, cp = sc[0]
            if cp.exists(): shutil.copy2(cp, out); return str(out) if out.exists() else None

        if sc:
            inps, dls = [], []
            for ms, cp in sc:
                if cp.exists() and cp.stat().st_size > 44:
                    inps.append(str(cp)); dls.append(ms)
            if inps:
                cmd = FF.merge_cmd(self.ff, inps, dls, str(out), sr, self.use_s, self.norm, HW_PROFILE.ff_threads)
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE, **FF._si())
                _reg_pid(proc.pid)
                await proc.communicate()
                _unreg_pid(proc.pid)
                if out.exists() and out.stat().st_size > 44: return str(out)

        if audio:
            valid = sorted([by_id[u] for u in audio if u in by_id], key=lambda x: x["ms"])
            if not valid: return None
            fb = tmpdir / "_fb"; fb.mkdir(exist_ok=True)
            inps, dls = [], []
            for it in valid:
                d = audio.get(it["id"])
                if not d: continue
                fp = fb / f"{it['id']}.wav"; fp.write_bytes(d)
                inps.append(str(fp)); dls.append(it["ms"])
            if inps:
                cmd = FF.merge_cmd(self.ff, inps, dls, str(out), sr, self.use_s, self.norm, HW_PROFILE.ff_threads)
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE, **FF._si())
                _reg_pid(proc.pid); await proc.communicate(); _unreg_pid(proc.pid)

        return str(out) if out.exists() and out.stat().st_size > 44 else None


# ═══════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

class Orchestrator:
    def __init__(self, uq: Optional[queue.Queue]):
        self.uq = uq
        self.stop = threading.Event()

    def _ui(self, msg): self.uq and self.uq.put(msg)

    def run(self, proj: Path, srt: Path, outdir: Path, voice: str):
        t0 = time.time()
        try:
            self._ui(("LOG", f"Engine: {proj.name} | {HW_PROFILE.summary()}", "sys"))
            items = parse_srt(srt)
            if not items: raise ValueError("SRT empty or invalid")
            self._ui(("LOG", f"Parsed {len(items)} items", "sys"))

            token = Auth.get()
            if not token: raise RuntimeError("Auth failed")
            self._ui(("LOG", "Auth OK", "ok"))

            out_file = outdir / f"TTS_{proj.name}.wav"

            async def _run():
                async with Pipeline() as p:
                    return await p.process(items, out_file, voice, token, self.uq, self.stop)

            result = asyncio.run(_run())
            elapsed = time.time() - t0

            if result and Path(result).exists():
                mb = Path(result).stat().st_size / (1024 * 1024)
                self._ui(("LOG", f"Done in {elapsed:.1f}s -> {Path(result).name} ({mb:.1f}MB)", "ok"))
                self._ui(("DONE", result, elapsed, len(items)))
            else:
                self._ui(("ERR", "No output produced", elapsed))
        except FileNotFoundError as e:
            self._ui(("ERR", f"Missing: {e}", time.time() - t0))
        except Exception as e:
            _log.exception("Job failed")
            self._ui(("ERR", str(e), time.time() - t0))


# ═══════════════════════════════════════════════════════════════
#  UI HTML
# ═══════════════════════════════════════════════════════════════

HTML_UI = "<h1>Loading...</h1>"
try:
    _ui_path = Path(__file__).parent / "Ui.html"
    if _ui_path.exists(): HTML_UI = _ui_path.read_text("utf-8")
except Exception: pass


# ═══════════════════════════════════════════════════════════════
#  WEBVIEW API
# ═══════════════════════════════════════════════════════════════

class API:
    def refresh_projects(self):   CURRENT_UI and CURRENT_UI.refresh_projects()
    def select_project(self, n):  CURRENT_UI and CURRENT_UI.select_project(n)
    def import_file(self):        CURRENT_UI and CURRENT_UI.handle_import()
    def open_project_folder(self):CURRENT_UI and CURRENT_UI.open_folder("project")
    def open_output_folder(self): CURRENT_UI and CURRENT_UI.open_folder("output")
    def choose_output_dir(self):  CURRENT_UI and CURRENT_UI.handle_output_dir()
    def clear_all(self):          CURRENT_UI and CURRENT_UI.clear_all()
    def start_processing(self):   CURRENT_UI and CURRENT_UI.start_job()
    def stop_processing(self):    CURRENT_UI and CURRENT_UI.stop_job()
    def get_settings(self):       return CURRENT_UI.get_settings() if CURRENT_UI else {}
    def set_voice_id(self, v):    return CURRENT_UI.set_voice(v) if CURRENT_UI else {"ok": False}
    def fast_login(self):         return CURRENT_UI.handle_login() if CURRENT_UI else {"ok": False}
    def clear_token(self):        Auth.clear(); return {"ok": True}
    def get_hw_info(self):
        h = HW_PROFILE
        return {"tier": h.tier, "cpu_cores": h.cores, "ram_total_gb": h.ram_gb,
                "optimal_rps": h.rps, "concurrency": h.concurrency, "conn_pool": h.conn_pool,
                "merge_batch": h.merge_batch, "merge_workers": h.merge_workers, "transport": h.transport}


class App:
    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.sel: Optional[str] = None
        self.win = None
        self.orch: Optional[Orchestrator] = None
        self.state = Settings.load()

    def _js(self, fn: str, *args):
        if self.win:
            try:
                a = ", ".join(json.dumps(x, ensure_ascii=False) for x in args)
                self.win.evaluate_js(f"window.{fn}({a})")
            except Exception: pass

    def get_settings(self) -> dict:
        h = HW_PROFILE
        return {"output_dir": str(self.state.output_dir), "user_voice_id": self.state.voice_id,
                "hw": {"tier": h.tier, "cpu_cores": h.cores, "ram_total_gb": h.ram_gb,
                       "optimal_rps": h.rps, "concurrency": h.concurrency, "conn_pool": h.conn_pool,
                       "merge_batch": h.merge_batch, "merge_workers": h.merge_workers, "transport": h.transport}}

    def set_voice(self, v: str) -> dict:
        self.state.voice_id = v.strip(); Settings.save(self.state)
        self._js("setVoiceCurrent", self.state.voice_id)
        return {"ok": True, "voice_id": self.state.voice_id}

    def handle_login(self) -> dict:
        tok = Auth.refresh()
        self._js("log", "Login " + ("OK" if tok else "failed"), "ok" if tok else "err")
        return {"ok": bool(tok)}

    def refresh_projects(self):
        projects = []
        try:
            for e in sorted(C.ROOT_DIR.iterdir()):
                if e.is_dir():
                    srt = e / "source.srt"
                    if srt.exists():
                        items = parse_srt(srt)
                        desc = f"{len(items)} lines"
                    else:
                        desc = "No SRT"
                    projects.append({"name": e.name, "desc": desc})
        except Exception: pass
        self._js("updateProjects", projects)

    def select_project(self, name: str):
        self.sel = name
        self._js("selectProject", name)
        self._js("setStatus", "READY", f"Project: {name}", "ok")

    def handle_import(self):
        res = self.win.create_file_dialog(webview.FileDialog.OPEN, file_types=("SRT Files (*.srt)",))
        if not res: return
        src = Path(str(res[0] if isinstance(res, (list, tuple)) else res))
        name = re.sub(r"[^\w]+", "_", src.stem).strip("_") or "PROJ"
        dest = C.ROOT_DIR / name; dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest / "source.srt")
        self.refresh_projects(); self.select_project(name)

    def open_folder(self, target: str):
        path = C.ROOT_DIR / self.sel if target == "project" and self.sel else self.state.output_dir
        try:
            if sys.platform == "win32": os.startfile(str(path))
            elif sys.platform == "darwin": subprocess.run(["open", str(path)], check=False)
            else: subprocess.run(["xdg-open", str(path)], check=False)
        except Exception: pass

    def handle_output_dir(self):
        res = self.win.create_file_dialog(webview.FileDialog.FOLDER, directory=str(self.state.output_dir))
        if not res: return
        self.state.output_dir = Path(str(res[0] if isinstance(res, (list, tuple)) else res))
        Settings.save(self.state); self._js("setOutputDir", str(self.state.output_dir))

    def clear_all(self):
        for e in C.ROOT_DIR.iterdir():
            if e.is_dir(): _rmtree(e)
        self.sel = None; self.refresh_projects(); self._js("selectProject", None)

    def start_job(self):
        if not self.sel: return
        proj = C.ROOT_DIR / self.sel; srt = proj / "source.srt"
        if not srt.exists(): self._js("log", "source.srt not found!", "err"); return
        self._js("setProcessing", True); self._js("setProgress", 0)
        self.orch = Orchestrator(self.q)
        threading.Thread(target=self.orch.run, args=(proj, srt, self.state.output_dir, self.state.voice_id), daemon=True).start()
        threading.Thread(target=self._monitor, daemon=True).start()

    def stop_job(self):
        if self.orch: self.orch.stop.set()

    def _monitor(self):
        last = 0.0
        while True:
            try: msg = self.q.get(timeout=0.1)
            except queue.Empty: continue
            k = msg[0]
            if k == "LOG":
                self._js("log", msg[1], msg[2] if len(msg) > 2 else "")
            elif k == "PROG":
                d = msg[1]
                done, total = d[0], d[1]
                fail = d[2] if len(d) > 2 else 0
                ips = d[3] if len(d) > 3 else 0
                rps = d[4] if len(d) > 4 else 0
                cbs = d[5] if len(d) > 5 else ""
                eta = d[6] if len(d) > 6 else 0
                now = time.time()
                if now - last >= 0.08 or done >= total:
                    extra = f" | CB:{cbs}" if cbs and cbs != "CLOSED" else ""
                    self._js("setStatus", "RUNNING",
                             f"TTS {done}/{total} | {ips:.1f}/s | Rate:{rps}/s | ETA {_fmt(int(eta))}{extra}", "ok")
                    self._js("setJobProgress", done, total, fail)
                    last = now
            elif k == "DONE":
                self._js("setProcessing", False); self._js("setProgress", 100)
                self._js("setStatus", "DONE", f"Saved: {Path(msg[1]).name}", "ok")
                self._js("onJobComplete", True); break
            elif k == "ERR":
                self._js("setStatus", "ERROR", str(msg[1]), "bad")
                self._js("setProcessing", False); self._js("log", f"Error: {msg[1]}", "err"); break


# ═══════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════

def run_app():
    global CURRENT_UI
    app = App(); CURRENT_UI = app
    storage = Path(tempfile.gettempdir()) / "TTS_Pro" / "webview"
    storage.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WEBVIEW2_USER_DATA_FOLDER", str(storage))
    app.win = webview.create_window(
        f"{C.APP_NAME} {C.VERSION}", html=HTML_UI, js_api=API(),
        width=1180, height=780, background_color="#f8f9fc")
    webview.start(private_mode=False, storage_path=str(storage))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_app()
