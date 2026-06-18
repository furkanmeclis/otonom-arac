#!/usr/bin/env python3
"""
Donkey Car simülasyon + tünel (zrok/ngrok) + LAN IP + QR launcher.

Proje köküne kopyalanınca çalışır (manage.py, simulationconfig.py, models/ ile aynı dizinde).
Ortam: .venv/bin/python, DONKEY_PYTHON veya aktif conda/venv; zrok/ngrok PATH'te olmalı.

Kullanım:
  python donkeyctl.py start                  # pist + bağlantı modu + model menüsü
  python donkeyctl.py start --model dilara   # model adıyla başlat
  python donkeyctl.py start --tunnel ngrok   # menüsüz ngrok
  python donkeyctl.py start --tunnel lan     # aynı WiFi, IP ile QR
  python donkeyctl.py start --tunnel lan --lan-ip 192.168.1.42
  python donkeyctl.py start --no-model       # AI olmadan manuel sürüş
  python donkeyctl.py start -y               # menüleri atla
  python donkeyctl.py models                 # models/ listesi
  python donkeyctl.py stop
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".donkeyctl"
SESSION_FILE = STATE_DIR / "session.json"
QR_FILE = STATE_DIR / "donkey_qr.png"
LOG_DIR = STATE_DIR / "logs"
NGROK_API = "http://127.0.0.1:4040/api/tunnels"
DEFAULT_PORT = 8887
DEFAULT_TUNNEL = os.environ.get("DONKEY_TUNNEL", "zrok")
PROJECT_CONFIG = os.environ.get("DONKEY_CONFIG", "simulationconfig.py")
FALLBACK_PYTHON = str(ROOT / ".venv" / "bin" / "python")
MODEL_EXTENSIONS = (".keras", ".h5", ".tflite", ".savedmodel", ".trt", ".pth")
MODEL_EXT_PRIORITY = {ext: idx for idx, ext in enumerate(MODEL_EXTENSIONS)}
MODEL_SKIP_FILES = frozenset({"database.json", "model_export_manifest.json"})
TunnelMode = Literal["zrok", "ngrok", "lan", "none"]

TUNNEL_CHOICES: list[tuple[TunnelMode, str]] = [
    ("zrok", "zrok (uzaktan)"),
    ("ngrok", "ngrok (uzaktan)"),
    ("lan", "Yerel ağ (IP / aynı WiFi)"),
    ("none", "Sadece bu bilgisayar (localhost)"),
]

# Resmi free tier limitleri (CLI/API kalan kotayı vermezse referans)
NGROK_FREE_LIMITS = {
    "data_out_gb": 1,
    "http_requests": 20_000,
    "online_endpoints": 3,
}
ZROK_FREE_LIMITS = {
    "daily_gb": 5,
    "max_shares": 50,
    "max_environments": 25,
}

TRACKS: list[tuple[str, str]] = [
    ("donkey-generated-roads-v0", "Generated Roads"),
    ("donkey-generated-track-v0", "Generated Track"),
    ("donkey-warehouse-v0", "Warehouse"),
    ("donkey-avc-sparkfun-v0", "AVC Sparkfun"),
    ("donkey-roboracingleague-track-v0", "Robo Racing League"),
    ("donkey-waveshare-v0", "Waveshare"),
    ("donkey-minimonaco-track-v0", "Mini Monaco"),
    ("donkey-warren-track-v0", "Warren Track"),
    ("donkey-thunderhill-track-v0", "Thunderhill"),
    ("donkey-circuit-launch-track-v0", "Circuit Launch"),
    ("donkey-mountain-track-v0", "Mountain Track"),
]

TRACK_IDS = {track_id for track_id, _ in TRACKS}


def ensure_state_dir() -> None:
    STATE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)


def load_session() -> dict[str, Any]:
    if not SESSION_FILE.exists():
        return {}
    try:
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_session(data: dict[str, Any]) -> None:
    ensure_state_dir()
    SESSION_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_python() -> str:
    """Proje kökündeki manage.py ile uyumlu Python yorumlayıcısını bul."""
    explicit = os.environ.get("DONKEY_PYTHON")
    if explicit and Path(explicit).exists():
        return explicit

    candidates: list[str] = []
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        candidates.append(str(venv_py))
    if Path(FALLBACK_PYTHON).exists():
        candidates.append(FALLBACK_PYTHON)
    if sys.executable:
        candidates.append(sys.executable)
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    seen: set[str] = set()
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if real in seen:
            continue
        seen.add(real)
        try:
            result = subprocess.run(
                [candidate, "-c", "import donkeycar, docopt"],
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                return candidate
        except OSError:
            continue

    if explicit:
        return explicit
    if Path(FALLBACK_PYTHON).exists():
        return FALLBACK_PYTHON
    if sys.executable:
        return sys.executable
    raise SystemExit(
        "Donkey Python bulunamadı. conda activate donkey veya DONKEY_PYTHON ayarlayın."
    )


def load_project_config() -> Any | None:
    """simulationconfig.py (veya DONKEY_CONFIG) yükle."""
    config_path = ROOT / PROJECT_CONFIG
    if not config_path.exists():
        for fallback in ("simulationconfig.py", "myconfig.py", "config.py"):
            candidate = ROOT / fallback
            if candidate.exists():
                config_path = candidate
                break
        else:
            return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("donkeyctl_config", config_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def get_models_dir() -> Path:
    """Proje kökündeki models/ klasörü (sürüş modelleri burada)."""
    return ROOT / "models"


def list_models() -> list[tuple[str, str]]:
    """(görünen_ad, köke_göre_yol) — aynı isimde birden fazla uzantı varsa .h5 tercih edilir."""
    models_dir = get_models_dir()
    if not models_dir.is_dir():
        return []

    best: dict[str, tuple[int, Path]] = {}
    for path in models_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in MODEL_SKIP_FILES:
            continue
        ext = path.suffix.lower()
        if ext not in MODEL_EXT_PRIORITY:
            continue
        stem = path.stem
        rank = MODEL_EXT_PRIORITY[ext]
        if stem not in best or rank < best[stem][0]:
            best[stem] = (rank, path)

    items: list[tuple[str, str]] = []
    for stem in sorted(best, key=str.lower):
        if stem.endswith("_fp16") and stem[: -len("_fp16")] in best:
            continue
        rel = best[stem][1].relative_to(ROOT).as_posix()
        items.append((stem, rel))
    return items


def resolve_model_path(model_arg: str) -> str:
    """CLI argümanını manage.py --model yoluna çevirir."""
    if model_arg.endswith(".json"):
        raise SystemExit(
            f"'{model_arg}' bir eğitim kaydı (database.json), sürüş modeli değil.\n"
            "`.h5` veya `.tflite` model seçin."
        )

    direct = Path(model_arg)
    if direct.is_file():
        rel = direct.relative_to(ROOT) if direct.is_relative_to(ROOT) else direct
        return validate_model_path(str(rel))

    rel = ROOT / model_arg
    if rel.is_file():
        return validate_model_path(rel.relative_to(ROOT).as_posix())

    models_dir = get_models_dir()
    for ext in MODEL_EXTENSIONS:
        candidate = models_dir / f"{model_arg}{ext}"
        if candidate.is_file():
            return validate_model_path(candidate.relative_to(ROOT).as_posix())

    raise SystemExit(
        f"Model bulunamadı: {model_arg}\n`donkeyctl models` ile listeyi görün."
    )


def validate_model_path(model: str | None) -> str | None:
    if not model:
        return None
    name = Path(model).name
    if name in MODEL_SKIP_FILES or name.endswith(".json"):
        return None
    path = ROOT / model
    if not path.is_file():
        return None
    ext = path.suffix.lower()
    if ext not in MODEL_EXT_PRIORITY:
        return None
    return model.replace("\\", "/")


def default_model_choice() -> str | None:
    session = load_session()
    validated = validate_model_path(session.get("model"))
    if validated:
        return validated
    models = list_models()
    return models[0][1] if models else None


def pick_model(interactive: bool, model_arg: str | None, *, no_model: bool = False) -> str | None:
    if no_model:
        return None
    if model_arg:
        return resolve_model_path(model_arg)

    models = list_models()
    default = default_model_choice()
    default_index = next(
        (idx for idx, (_, path) in enumerate(models) if path == default),
        0 if models else None,
    )

    if not interactive or not sys.stdin.isatty():
        if default:
            label = next((name for name, path in models if path == default), default)
            print(f"Model (varsayılan): {label} ({default})")
            return default
        print("Model: yok (manuel sürüş)")
        return None

    print("\nModel seçin:\n")
    for idx, (name, rel_path) in enumerate(models, start=1):
        marker = " ← varsayılan" if default_index is not None and idx - 1 == default_index else ""
        print(f"  {idx:2}. {name:<24} ({rel_path}){marker}")
    manual_idx = len(models) + 1
    print(f"  {manual_idx:2}. Manuel sürüş (AI modeli yok)")
    print()

    while True:
        default_choice = (default_index + 1) if default_index is not None else manual_idx
        choice = input(
            f"Seçim [1-{manual_idx}] (Enter={default_choice}): "
        ).strip()
        if choice == "":
            if default_index is not None:
                return models[default_index][1]
            return None
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(models):
                return models[index][1]
            if index == len(models):
                return None
        print("Geçersiz seçim, tekrar deneyin.")


def resolve_zrok2() -> str | None:
    home_bin = Path.home() / "bin" / "zrok2"
    if home_bin.exists():
        return str(home_bin)
    return shutil.which("zrok2")


def pick_track(interactive: bool, track_arg: str | None) -> str:
    if track_arg:
        if track_arg not in TRACK_IDS:
            raise SystemExit(f"Bilinmeyen pist: {track_arg}\n`donkeyctl tracks` ile listeyi görün.")
        return track_arg

    if not interactive or not sys.stdin.isatty():
        session = load_session()
        if session.get("track"):
            print(f"Pist (session): {session['track']}")
            return session["track"]
        return TRACKS[0][0]

    print("\nPist seçin:\n")
    for idx, (track_id, label) in enumerate(TRACKS, start=1):
        print(f"  {idx:2}. {label:<28} ({track_id})")
    print()

    while True:
        choice = input(f"Seçim [1-{len(TRACKS)}] (Enter=1): ").strip()
        if choice == "":
            return TRACKS[0][0]
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(TRACKS):
                return TRACKS[index][0]
        print("Geçersiz seçim, tekrar deneyin.")


def read_ngrok_config() -> dict[str, Any]:
    paths = [
        Path.home() / "Library/Application Support/ngrok/ngrok.yml",
        Path.home() / ".config/ngrok/ngrok.yml",
        Path.home() / ".ngrok2/ngrok.yml",
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            import yaml  # type: ignore[import-untyped]

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # yaml yoksa basit satir arama
        text = path.read_text(encoding="utf-8", errors="ignore")
        cfg: dict[str, Any] = {}
        for line in text.splitlines():
            if "api_key:" in line:
                cfg["api_key"] = line.split(":", 1)[1].strip().strip('"').strip("'")
            if "authtoken:" in line:
                cfg["authtoken"] = line.split(":", 1)[1].strip().strip('"').strip("'")
        if cfg:
            return cfg
    return {}


def get_zrok_tunnel_info() -> dict[str, Any]:
    zrok2 = resolve_zrok2()
    if not zrok2:
        return {"installed": False, "enabled": False, "share_count": 0}

    result = subprocess.run([zrok2, "status"], capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    enabled = "Account Token" in output and "<<SET>>" in output
    share_count = 0
    if enabled:
        overview = zrok_overview_json()
        for env in overview.get("environments", []):
            share_count += len(env.get("shares", []))

    return {
        "installed": True,
        "enabled": enabled,
        "share_count": share_count,
    }


def get_ngrok_tunnel_info() -> dict[str, Any]:
    installed = shutil.which("ngrok") is not None
    cfg = read_ngrok_config()
    agent_cfg = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}
    authtoken = cfg.get("api_key") or cfg.get("authtoken") or agent_cfg.get("authtoken")
    api_key = cfg.get("api_key") or agent_cfg.get("api_key")

    info: dict[str, Any] = {
        "installed": installed,
        "configured": bool(authtoken),
        "has_api_key": bool(api_key),
        "agent_online": False,
        "online_tunnels": 0,
    }

    try:
        with urllib.request.urlopen(NGROK_API, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        info["agent_online"] = True
        info["online_tunnels"] = len(data.get("tunnels", []))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        pass

    return info


def _is_valid_lan_ipv4(ip: str) -> bool:
    try:
        addr = ipaddress.IPv4Address(ip)
        return not (addr.is_loopback or addr.is_link_local or addr.is_multicast)
    except ipaddress.AddressValueError:
        return False


def get_lan_ipv4_candidates() -> list[str]:
    """UDP egress + hostname çözümlemesi; loopback ve link-local filtrelenir."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(ip: str) -> None:
        if _is_valid_lan_ipv4(ip) and ip not in seen:
            seen.add(ip)
            candidates.append(ip)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            add(sock.getsockname()[0])
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass

    return candidates


def detect_primary_lan_ip() -> str | None:
    candidates = get_lan_ipv4_candidates()
    return candidates[0] if candidates else None


def resolve_lan_base_url(port: int, lan_ip: str) -> str:
    return f"http://{lan_ip}:{port}"


def pick_lan_ip(interactive: bool, lan_ip_arg: str | None = None) -> str:
    if lan_ip_arg:
        if not _is_valid_lan_ipv4(lan_ip_arg):
            raise SystemExit(f"Geçersiz LAN IP: {lan_ip_arg}")
        return lan_ip_arg

    session = load_session()
    saved = session.get("lan_ip")
    if saved and _is_valid_lan_ipv4(str(saved)):
        candidates = get_lan_ipv4_candidates()
        if str(saved) in candidates or not interactive:
            return str(saved)

    candidates = get_lan_ipv4_candidates()
    if not candidates:
        raise SystemExit(
            "LAN IPv4 adresi bulunamadı. WiFi'ye bağlı olduğunuzdan emin olun\n"
            "veya manuel IP verin: --lan-ip 192.168.x.x"
        )

    if len(candidates) == 1 or not interactive or not sys.stdin.isatty():
        return candidates[0]

    print("\nYerel ağ IP seçin:\n")
    for idx, ip in enumerate(candidates, start=1):
        print(f"  {idx}. {ip}")
    while True:
        choice = input(f"Seçim [1-{len(candidates)}] (Enter=1): ").strip()
        if choice == "":
            return candidates[0]
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(candidates):
                return candidates[index]
        print("Geçersiz seçim, tekrar deneyin.")


def tunnel_quota_hint(mode: TunnelMode) -> str:
    if mode == "zrok":
        info = get_zrok_tunnel_info()
        parts: list[str] = []
        if info["installed"]:
            parts.append("✓ kurulu")
            parts.append("✓ aktif" if info["enabled"] else "✗ zrok2 enable gerekli")
            if info["enabled"]:
                parts.append(f"{info['share_count']} aktif share")
        else:
            parts.append("✗ kurulu değil (~/bin/zrok2)")
        parts.append(
            f"Free: {ZROK_FREE_LIMITS['daily_gb']} GB/gün "
            f"(kalan kota CLI'dan okunamaz → myzrok.io)"
        )
        return " · ".join(parts)

    if mode == "ngrok":
        info = get_ngrok_tunnel_info()
        parts = []
        if info["installed"]:
            parts.append("✓ kurulu")
            parts.append("✓ token" if info["configured"] else "✗ authtoken yok")
            if info["agent_online"]:
                parts.append(
                    f"{info['online_tunnels']}/{NGROK_FREE_LIMITS['online_endpoints']} online endpoint"
                )
            if info["has_api_key"]:
                parts.append("API key var (dashboard ile senkron)")
            else:
                parts.append("kalan kota: dashboard.ngrok.com/usage")
        else:
            parts.append("✗ kurulu değil (brew install ngrok)")
        parts.append(
            f"Free: {NGROK_FREE_LIMITS['data_out_gb']} GB/ay çıkış, "
            f"{NGROK_FREE_LIMITS['http_requests']:,} HTTP/ay"
        )
        return " · ".join(parts)

    if mode == "lan":
        ip = detect_primary_lan_ip()
        hint = f"algılanan IP: {ip}" if ip else "IP algılanamadı"
        return f"{hint} · kota yok · aynı WiFi gerekli"

    return "Public URL yok · sadece localhost:8887"


def default_tunnel_choice() -> TunnelMode:
    session = load_session()
    tunnel = session.get("tunnel")
    if tunnel == "cloudflare":
        tunnel = "zrok"
    if tunnel in ("zrok", "ngrok", "lan", "none"):
        return tunnel  # type: ignore[return-value]
    default = os.environ.get("DONKEY_TUNNEL", DEFAULT_TUNNEL)
    if default in ("zrok", "ngrok", "lan", "none"):
        return default  # type: ignore[return-value]
    return "zrok"


def pick_tunnel(interactive: bool, tunnel_arg: TunnelMode | None) -> TunnelMode:
    if tunnel_arg is not None:
        return tunnel_arg

    default = default_tunnel_choice()
    default_index = next(i for i, (mode, _) in enumerate(TUNNEL_CHOICES) if mode == default)

    if not interactive or not sys.stdin.isatty():
        print(f"Bağlantı modu (varsayılan): {TUNNEL_CHOICES[default_index][1]} ({default})")
        return default

    print("\nBağlantı modu seçin:\n")
    for idx, (mode, label) in enumerate(TUNNEL_CHOICES, start=1):
        marker = " ← varsayılan" if idx - 1 == default_index else ""
        print(f"  {idx}. {label}{marker}")
        print(f"      {tunnel_quota_hint(mode)}")
    print("\n  Kota detayı: ngrok → https://dashboard.ngrok.com/usage")
    print("               zrok → https://myzrok.io\n")

    while True:
        choice = input(
            f"Seçim [1-{len(TUNNEL_CHOICES)}] (Enter={default_index + 1}): "
        ).strip()
        if choice == "":
            return TUNNEL_CHOICES[default_index][0]
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(TUNNEL_CHOICES):
                return TUNNEL_CHOICES[index][0]
        print("Geçersiz seçim, tekrar deneyin.")


def is_pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_sim_path() -> str | None:
    mod = load_project_config()
    if mod is None:
        return None
    try:
        path = getattr(mod, "DONKEY_SIM_PATH", None)
        if path and path != "remote" and Path(path).exists():
            return str(path)
    except Exception:
        pass
    return None


def stop_simulator(sim_path: str | None = None) -> int:
    sim_path = sim_path or load_session().get("sim_path") or get_sim_path()
    patterns = [
        "Contents/MacOS/donkey_sim",
        "DonkeySimLinux/donkey_sim",
        "donkey_sim.x86_64",
    ]
    if sim_path:
        patterns.insert(0, sim_path)

    pids: set[int] = set()
    for pattern in patterns:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue
        for line in result.stdout.splitlines():
            if line.strip().isdigit():
                pids.add(int(line.strip()))

    if not pids:
        return 0

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    time.sleep(1.0)
    for pid in list(pids):
        if is_pid_running(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    return len(pids)


def zrok_overview_json() -> dict[str, Any]:
    zrok2 = resolve_zrok2()
    if not zrok2:
        return {}
    result = subprocess.run(
        [zrok2, "overview", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def find_zrok_share(target: str) -> dict[str, Any] | None:
    data = zrok_overview_json()
    for env in data.get("environments", []):
        for share in env.get("shares", []):
            if share.get("target") == target:
                return share
    return None


def zrok_share_to_public_url(share: dict[str, Any]) -> str | None:
    endpoints = share.get("frontendEndpoints") or []
    if not endpoints:
        return None
    endpoint = endpoints[0]
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint.rstrip("/")
    return f"https://{endpoint}".rstrip("/")


def ensure_zrok_enabled() -> None:
    zrok2 = resolve_zrok2()
    if not zrok2:
        raise SystemExit(
            "zrok2 bulunamadi. Kurulum: ~/bin/zrok2 veya PATH'e ekle"
        )
    result = subprocess.run([zrok2, "status"], capture_output=True, text=True)
    output = result.stdout + result.stderr
    if "Account Token" not in output or "<<SET>>" not in output:
        raise SystemExit(
            "zrok2 ortami aktif degil. Once calistir:\n  zrok2 enable"
        )


def stop_zrok_tunnel(share_token: str | None = None) -> None:
    zrok2 = resolve_zrok2()
    if not zrok2:
        return

    session = load_session()
    token = share_token or session.get("zrok_share_token")
    if token:
        subprocess.run([zrok2, "delete", "share", token], check=False)
        time.sleep(2.0)

    subprocess.run(["pkill", "-f", "zrok2 share public"], check=False)


def stop_ngrok_tunnel() -> None:
    subprocess.run(["pkill", "-f", f"ngrok http {DEFAULT_PORT}"], check=False)


def stop_orphan_cloudflared() -> None:
    """Sistemde kalmış cloudflared süreçlerini kapat (502/bad gateway kaynağı)."""
    subprocess.run(["pkill", "-f", "cloudflared tunnel"], check=False)


def is_http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def wait_for_public_http(
    public_url: str,
    *,
    local_url: str | None = None,
    timeout: float = 60.0,
) -> bool:
    """Tünel üzerinden /drive erişimini tekrarlı dener (zrok proxy gecikmesi için)."""
    target = f"{public_url.rstrip('/')}/drive"
    deadline = time.time() + timeout
    printed = False
    while time.time() < deadline:
        if local_url and not is_http_ok(local_url):
            time.sleep(1)
            continue
        if is_http_ok(target, timeout=8.0):
            return True
        if not printed:
            print("Public URL test ediliyor (zrok proxy hazırlanıyor)...")
            printed = True
        time.sleep(2)
    return False


def rollback_failed_start(
    manage_proc: subprocess.Popen[Any],
    tunnel_proc: subprocess.Popen[Any] | None,
    tunnel: TunnelMode,
    zrok_share_token: str | None,
) -> None:
    if tunnel_proc and is_pid_running(tunnel_proc.pid):
        try:
            tunnel_proc.terminate()
        except OSError:
            pass
    if is_pid_running(manage_proc.pid):
        try:
            manage_proc.terminate()
        except OSError:
            pass
    stop_tunnel({"tunnel": tunnel, "zrok_share_token": zrok_share_token})
    stop_simulator()
    subprocess.run(["pkill", "-f", "manage.py drive"], check=False)


def stop_tunnel(session: dict[str, Any]) -> None:
    tunnel = session.get("tunnel", "ngrok")
    tunnel_pid = session.get("tunnel_pid") or session.get("ngrok_pid")

    if is_pid_running(tunnel_pid):
        try:
            os.kill(tunnel_pid, signal.SIGTERM)
        except OSError:
            pass

    if tunnel == "zrok":
        stop_zrok_tunnel(session.get("zrok_share_token"))
    elif tunnel == "ngrok":
        stop_ngrok_tunnel()
    stop_orphan_cloudflared()


def stop_processes() -> None:
    session = load_session()

    manage_pid = session.get("manage_pid")
    if is_pid_running(manage_pid):
        try:
            os.kill(manage_pid, signal.SIGTERM)
        except OSError:
            pass
        time.sleep(1.5)
        if is_pid_running(manage_pid):
            try:
                os.kill(manage_pid, signal.SIGKILL)
            except OSError:
                pass

    stop_tunnel(session)
    subprocess.run(["pkill", "-f", "manage.py drive"], check=False)

    sim_killed = stop_simulator(session.get("sim_path"))

    for _ in range(20):
        if not _port_in_use(DEFAULT_PORT):
            break
        time.sleep(0.25)

    session["manage_pid"] = None
    session["tunnel_pid"] = None
    session["ngrok_pid"] = None
    session["zrok_share_token"] = None
    session["running"] = False
    save_session(session)

    parts = ["manage.py", session.get("tunnel", "tünel")]
    if sim_killed:
        parts.append("simülasyon")
    print(f"Servisler durduruldu ({' + '.join(parts)}).")


def _port_in_use(port: int) -> bool:
    try:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def wait_for_http(url: str, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    return False


def wait_for_ngrok_url(timeout: float = 30.0) -> str | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(NGROK_API, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for tunnel in data.get("tunnels", []):
                if tunnel.get("proto") == "https":
                    return tunnel["public_url"]
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            pass
        time.sleep(1)
    return None


def wait_for_zrok_url(port: int, timeout: float = 45.0) -> tuple[str | None, str | None]:
    """Public URL ve shareToken doner."""
    target = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        share = find_zrok_share(target)
        if share:
            url = zrok_share_to_public_url(share)
            token = share.get("shareToken")
            if url:
                return url, token
        time.sleep(1)
    return None, None


def start_zrok_tunnel(port: int) -> tuple[subprocess.Popen[Any], str, str]:
    ensure_zrok_enabled()
    zrok2 = resolve_zrok2()
    assert zrok2 is not None

    target = f"http://127.0.0.1:{port}"
    zrok_log = open(LOG_DIR / "zrok.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        [zrok2, "share", "public", target, "--headless"],
        stdout=zrok_log,
        stderr=subprocess.STDOUT,
    )

    print("zrok tüneli açılıyor...")
    public_url, share_token = wait_for_zrok_url(port)
    if not public_url or not share_token:
        proc.terminate()
        raise SystemExit(f"zrok URL alınamadı. Log: {LOG_DIR / 'zrok.log'}")

    time.sleep(2.0)
    return proc, public_url, share_token


def start_ngrok_tunnel(port: int) -> tuple[subprocess.Popen[Any], str]:
    if shutil.which("ngrok") is None:
        raise SystemExit("ngrok bulunamadı. Kur: brew install ngrok")

    ngrok_log = open(LOG_DIR / "ngrok.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        ["ngrok", "http", str(port), "--log=stdout"],
        stdout=ngrok_log,
        stderr=subprocess.STDOUT,
    )
    print("ngrok tüneli açılıyor...")
    public_url = wait_for_ngrok_url()
    if not public_url:
        proc.terminate()
        raise SystemExit(f"ngrok URL alınamadı. Log: {LOG_DIR / 'ngrok.log'}")
    return proc, public_url


def build_connection_payload(
    base_url: str,
    track: str,
    tunnel: str,
    model: str | None = None,
) -> dict[str, Any]:
    ws_base = base_url.replace("https://", "wss://").replace("http://", "ws://")
    payload: dict[str, Any] = {
        "v": 1,
        "base": base_url.rstrip("/"),
        "track": track,
        "tunnel": tunnel,
        "ws": f"{ws_base.rstrip('/')}/wsDrive",
        "video": f"{base_url.rstrip('/')}/video",
        "snapshot": f"{base_url.rstrip('/')}/snapshot",
        "drive": f"{base_url.rstrip('/')}/drive",
        "port": DEFAULT_PORT,
    }
    if model:
        payload["model"] = model
    return payload


def show_qr(payload: dict[str, Any], python_bin: str | None = None) -> None:
    """QR kodu terminale basar (Expo tarzı); PNG dosyası açılmaz."""
    ensure_state_dir()
    payload_json = json.dumps(payload, ensure_ascii=False)
    python_bin = python_bin or resolve_python()

    def _print_ascii(data: str) -> bool:
        try:
            import qrcode

            qr = qrcode.QRCode(border=1, box_size=1)
            qr.add_data(data)
            qr.make(fit=True)
            print("\nMobil cihazdan QR tarayın:\n")
            qr.print_ascii(invert=True)
            print()
            return True
        except ImportError:
            return False

    if _print_ascii(payload_json):
        try:
            import qrcode

            qrcode.make(payload_json).save(QR_FILE)
            print(f"QR yedek dosya: {QR_FILE}")
        except Exception:
            pass
        print(f"Payload:\n{payload_json}\n")
        return

    script = f"""
import qrcode
payload = {json.dumps(payload_json)!r}
qr = qrcode.QRCode(border=1, box_size=1)
qr.add_data(payload)
qr.make(fit=True)
print()
print('Mobil cihazdan QR tarayın:')
print()
qr.print_ascii(invert=True)
print()
qrcode.make(payload).save({str(QR_FILE)!r})
print('QR yedek dosya:', {str(QR_FILE)!r})
"""
    result = subprocess.run([python_bin, "-c", script], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        print(f"Payload:\n{payload_json}\n")
        return

    print(f"\n[qrcode] paketi yok ({python_bin}). Kurulum:")
    print(f"  {python_bin} -m pip install qrcode")
    print(f"\nPayload (mobil app'e yapıştır):\n{payload_json}\n")


def print_connection_info(
    payload: dict[str, Any],
    local_only: bool = False,
    *,
    lan_ip: str | None = None,
) -> None:
    tunnel = payload.get("tunnel", "zrok")
    port = payload.get("port", DEFAULT_PORT)
    print("\n=== Bağlantı Bilgileri ===")
    print(f"Pist      : {payload.get('track')}")
    if payload.get("model"):
        print(f"Model     : {payload.get('model')}")
    print(f"Mod       : {tunnel}")
    print(f"Port      : {port}")
    print(f"Lokal UI  : http://localhost:{port}/drive")
    print(f"Lokal WSS : ws://localhost:{port}/wsDrive")
    if tunnel == "lan":
        if lan_ip:
            print(f"LAN IP    : {lan_ip}")
        print(f"LAN base  : {payload.get('base')}")
        print(f"Mobil WSS : {payload.get('ws')}")
        print(f"Video     : {payload.get('video')}")
        print(f"Snapshot  : {payload.get('snapshot')}  ← RN icin onerilen")
        print("Not       : Aynı WiFi gerekli; güvenlik duvarı engelleyebilir")
    elif not local_only:
        print(f"Public    : {payload.get('base')}")
        print(f"Mobil WSS : {payload.get('ws')}")
        print(f"Video     : {payload.get('video')}")
        print(f"Snapshot  : {payload.get('snapshot')}  ← RN icin onerilen")
        if tunnel == "ngrok":
            print(f"ngrok UI  : http://127.0.0.1:4040")
        elif tunnel == "zrok":
            print(f"zrok      : zrok2 overview")
    print("==========================\n")


def start_services(
    track: str,
    *,
    tunnel: TunnelMode = "zrok",
    model: str | None = None,
    record: bool = False,
    port: int = DEFAULT_PORT,
    lan_ip: str | None = None,
    interactive: bool = False,
) -> None:
    ensure_state_dir()
    python_bin = resolve_python()

    if not Path(python_bin).exists():
        raise SystemExit(
            f"Python bulunamadı: {python_bin}\n"
            "conda activate donkey veya DONKEY_PYTHON ayarlayın."
        )

    if tunnel == "zrok" and resolve_zrok2() is None:
        raise SystemExit("zrok2 bulunamadi. Manuel kurulum: ~/bin/zrok2")

    stop_orphan_cloudflared()
    stop_processes()

    model = validate_model_path(model)

    env = os.environ.copy()
    env["DONKEY_GYM_ENV_NAME"] = track
    env["AUTO_RECORD_ON_THROTTLE"] = "true" if record else "false"
    env["WEB_CONTROL_PORT"] = str(port)
    if "DONKEY_JPEG_QUALITY" not in env:
        env["DONKEY_JPEG_QUALITY"] = os.environ.get("DONKEY_JPEG_QUALITY", "90")

    manage_cmd = [
        python_bin,
        "manage.py",
        "drive",
        f"--simulationconfig={PROJECT_CONFIG}",
        "--type=target_point",
    ]
    if model:
        manage_cmd.append(f"--model={model}")

    manage_log = open(LOG_DIR / "manage.log", "a", encoding="utf-8")
    manage_proc = subprocess.Popen(
        manage_cmd,
        cwd=ROOT,
        env=env,
        stdout=manage_log,
        stderr=subprocess.STDOUT,
    )

    print("Simülasyon başlatılıyor...")
    if model:
        print(f"AI model  : {model}")
    local_url = f"http://127.0.0.1:{port}/drive"
    if not wait_for_http(local_url):
        manage_proc.terminate()
        raise SystemExit(f"manage.py {port} portunda ayağa kalkmadı. Log: {LOG_DIR / 'manage.log'}")

    tunnel_proc = None
    public_url = f"http://127.0.0.1:{port}"
    zrok_share_token = None
    resolved_lan_ip: str | None = None

    if tunnel == "zrok":
        tunnel_proc, public_url, zrok_share_token = start_zrok_tunnel(port)
    elif tunnel == "ngrok":
        tunnel_proc, public_url = start_ngrok_tunnel(port)
    elif tunnel == "lan":
        resolved_lan_ip = pick_lan_ip(interactive=interactive, lan_ip_arg=lan_ip)
        public_url = resolve_lan_base_url(port, resolved_lan_ip)
        lan_drive = f"{public_url}/drive"
        print(f"LAN IP    : {resolved_lan_ip}")
        if not wait_for_http(lan_drive, timeout=15.0):
            rollback_failed_start(manage_proc, tunnel_proc, tunnel, zrok_share_token)
            raise SystemExit(
                f"LAN URL yanıt vermiyor: {lan_drive}\n"
                f"Lokal: {local_url}\n"
                "Olası nedenler:\n"
                "  - macOS Güvenlik Duvarı Python gelen bağlantıları engelliyor\n"
                "  - Yanlış IP (--lan-ip ile deneyin)\n"
                "  - Telefon farklı ağda (aynı WiFi gerekli)"
            )

    if tunnel in ("zrok", "ngrok") and not wait_for_public_http(
        public_url, local_url=local_url, timeout=60.0
    ):
        rollback_failed_start(manage_proc, tunnel_proc, tunnel, zrok_share_token)
        raise SystemExit(
            f"Tünel açıldı ama public URL yanıt vermiyor (502/bad gateway):\n"
            f"  {public_url}/drive\n"
            f"Lokal: {local_url}\n"
            "Servisler durduruldu. Tekrar dene:\n"
            "  python donkeyctl.py start"
        )

    payload = build_connection_payload(public_url, track, tunnel, model=model)
    sim_path = get_sim_path()
    session = {
        "running": True,
        "track": track,
        "model": model,
        "record": record,
        "port": port,
        "sim_path": sim_path,
        "tunnel": tunnel,
        "lan_ip": resolved_lan_ip,
        "manage_pid": manage_proc.pid,
        "tunnel_pid": tunnel_proc.pid if tunnel_proc else None,
        "ngrok_pid": tunnel_proc.pid if tunnel == "ngrok" and tunnel_proc else None,
        "zrok_share_token": zrok_share_token,
        "use_ngrok": tunnel == "ngrok",
        "connection": payload,
        "started_at": time.time(),
    }
    save_session(session)

    print_connection_info(
        payload,
        local_only=(tunnel == "none"),
        lan_ip=resolved_lan_ip,
    )
    show_qr(payload, python_bin=python_bin)
    print("Ctrl+C ile durdurmak için: python donkeyctl.py stop")


def cmd_status(_: argparse.Namespace) -> None:
    session = load_session()
    payload = session.get("connection")
    if not payload:
        print("Aktif session yok. `python donkeyctl.py start` çalıştırın.")
        return

    manage_running = is_pid_running(session.get("manage_pid"))
    tunnel_running = is_pid_running(session.get("tunnel_pid") or session.get("ngrok_pid"))
    port_open = _port_in_use(session.get("port", DEFAULT_PORT))
    tunnel = session.get("tunnel", "ngrok" if session.get("use_ngrok") else "none")

    print("\n=== Durum ===")
    print(f"manage.py : {'çalışıyor' if manage_running else 'kapalı'}")
    print(f"mod       : {tunnel} ({'çalışıyor' if tunnel_running else 'kapalı'})")
    print(f"Port      : {'açık' if port_open else 'kapalı'}")
    if tunnel == "lan" and session.get("lan_ip"):
        print(f"LAN IP    : {session.get('lan_ip')}")
    if tunnel in ("zrok", "ngrok"):
        print(f"kota      : {tunnel_quota_hint(tunnel)}")
    print_connection_info(
        payload,
        local_only=(tunnel == "none"),
        lan_ip=session.get("lan_ip"),
    )


def cmd_qr(_: argparse.Namespace) -> None:
    session = load_session()
    payload = session.get("connection")
    if not payload:
        print("Session yok. Önce `start` veya `restart` çalıştırın.")
        return
    show_qr(payload)


def cmd_models(_: argparse.Namespace) -> None:
    print("\nKullanılabilir modeller:\n")
    models = list_models()
    if not models:
        print(f"  (models/ boş — {get_models_dir()})")
    else:
        for name, rel_path in models:
            print(f"  {name:<24}  {rel_path}")
    print()


def cmd_tracks(_: argparse.Namespace) -> None:
    print("\nKullanılabilir pistler:\n")
    for track_id, label in TRACKS:
        print(f"  {label:<28}  {track_id}")
    print()


def parse_tunnel(value: str) -> TunnelMode:
    if value == "cloudflare":
        print("Not: cloudflare kaldırıldı, zrok kullanılıyor.")
        return "zrok"
    if value not in ("zrok", "ngrok", "lan", "none"):
        raise argparse.ArgumentTypeError("tunnel: zrok, ngrok, lan veya none olmali")
    return value  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Donkey Car simülasyon launcher (sim + zrok/ngrok/lan + QR)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Simülasyonu başlat")
    start.add_argument("--track", help="Pist ID (ör. donkey-warren-track-v0)")
    start.add_argument(
        "--tunnel",
        type=parse_tunnel,
        default=None,
        help="Bağlantı modu: zrok, ngrok, lan, none (verilmezse menüden seçilir)",
    )
    start.add_argument(
        "--lan-ip",
        help="LAN modu için manuel IPv4 (ör. 192.168.1.42)",
    )
    start.add_argument("--no-ngrok", action="store_true", help="--tunnel none ile ayni")
    start.add_argument("--model", help="Model yolu veya adı (ör. my_first_pilot)")
    start.add_argument("--no-model", action="store_true", help="AI modeli yüklemeden başlat")
    start.add_argument("--record", action="store_true", help="Otomatik kaydı aç")
    start.add_argument("--port", type=int, default=DEFAULT_PORT)
    start.add_argument("-y", "--yes", action="store_true", help="Pist, tünel ve model seçimini atla")

    restart = sub.add_parser("restart", help="Hard refresh — aynı ayarlarla yeniden başlat")
    restart.add_argument(
        "--tunnel",
        type=parse_tunnel,
        help="Tüneli override et (yoksa session'dan)",
    )

    restart.add_argument(
        "--lan-ip",
        help="LAN modu için manuel IPv4 override",
    )

    sub.add_parser("stop", help="Tüm servisleri durdur")
    sub.add_parser("status", help="Bağlantı bilgilerini göster")
    sub.add_parser("qr", help="QR kodunu terminale yeniden yazdır")
    sub.add_parser("tracks", help="Pist listesi")
    sub.add_parser("models", help="Model listesi")

    return parser


def ensure_donkey_python() -> None:
    donkey_py = resolve_python()
    if not Path(donkey_py).exists():
        return
    if os.path.realpath(sys.executable) != os.path.realpath(donkey_py):
        os.execv(donkey_py, [donkey_py, *sys.argv])


def session_tunnel(session: dict[str, Any]) -> TunnelMode:
    tunnel = session.get("tunnel")
    if tunnel == "cloudflare":
        return "zrok"
    if tunnel in ("zrok", "ngrok", "lan", "none"):
        return tunnel
    if session.get("use_ngrok"):
        return "ngrok"
    return "none"


def main() -> None:
    ensure_donkey_python()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "start":
        track = pick_track(interactive=not args.yes, track_arg=args.track)
        if args.no_ngrok:
            tunnel: TunnelMode = "none"
        else:
            tunnel = pick_tunnel(interactive=not args.yes, tunnel_arg=args.tunnel)
        model = pick_model(
            interactive=not args.yes,
            model_arg=args.model,
            no_model=args.no_model,
        )
        start_services(
            track,
            tunnel=tunnel,
            model=model,
            record=args.record,
            port=args.port,
            lan_ip=args.lan_ip,
            interactive=not args.yes,
        )
    elif args.command == "stop":
        stop_processes()
    elif args.command == "restart":
        session = load_session()
        tunnel = args.tunnel if getattr(args, "tunnel", None) else None
        if not session:
            print("Önceki session yok, varsayılan ayarlarla başlatılıyor...")
            track = pick_track(interactive=True, track_arg=None)
            start_services(track, tunnel=parse_tunnel(DEFAULT_TUNNEL), record=False)
        else:
            print("Hard refresh...")
            start_services(
                session.get("track", TRACKS[0][0]),
                tunnel=tunnel or session_tunnel(session),
                model=session.get("model"),
                record=session.get("record", False),
                port=session.get("port", DEFAULT_PORT),
                lan_ip=getattr(args, "lan_ip", None) or session.get("lan_ip"),
                interactive=False,
            )
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "qr":
        cmd_qr(args)
    elif args.command == "tracks":
        cmd_tracks(args)
    elif args.command == "models":
        cmd_models(args)


if __name__ == "__main__":
    main()
