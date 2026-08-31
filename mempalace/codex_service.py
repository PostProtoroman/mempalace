"""Background-service lifecycle for the Codex capture follower.

``mempalace codex-capture`` in the foreground dies with the terminal, which
is not a capture story anyone can rely on. This module installs the follower
as a per-user login service so it survives terminal exit and comes back at
login, while staying entirely local: a loopback-only follower, a plist under
the user's own LaunchAgents, no network exposure and no root.

macOS uses launchd, following the same shape as the palace hub's own
``ai.mempalace.hub`` agent (``RunAtLoad`` + ``KeepAlive``). Other platforms
report unsupported rather than pretending — a half-working service is worse
than an honest refusal.

Every path is derived from ``home``, which defaults to the real home but is
overridable. Tests pass a temporary home, so installing and uninstalling a
service never touches the machine's real launchd state.

Uninstall removes only files this module created — the plist and the log
files under the palace's ``codex_capture`` state dir. Palace data and any
user-authored content are never touched.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

LABEL = "ai.mempalace.codex-capture"


class ServiceError(RuntimeError):
    """Raised when a lifecycle action cannot be completed."""


def is_supported() -> bool:
    return sys.platform == "darwin"


def _home(home: Optional[str] = None) -> Path:
    return Path(home).expanduser() if home else Path.home()


def plist_path(home: Optional[str] = None) -> Path:
    return _home(home) / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def log_dir(palace_path: str) -> Path:
    from .codex_follower import state_dir

    return state_dir(palace_path)


def _launchctl(*args: str, home: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _domain() -> str:
    return f"gui/{os.getuid()}"


def build_plist(
    palace_path: str,
    sessions_root: str,
    *,
    interval: float = 15.0,
    executable: Optional[str] = None,
    home: Optional[str] = None,
) -> dict:
    """The launchd job definition. Pure — no filesystem writes."""
    logs = log_dir(palace_path)
    program = [
        executable or sys.executable,
        "-m",
        "mempalace",
        "codex-capture",
        "--palace",
        str(palace_path),
        "--sessions-root",
        str(sessions_root),
        "--interval",
        str(interval),
        # The service must not rewrite the user's AGENTS.md on every login;
        # recall installation is an explicit, separate step.
        "--no-recall",
    ]
    return {
        "Label": LABEL,
        "ProgramArguments": program,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Background",
        "StandardOutPath": str(logs / "service.out.log"),
        "StandardErrorPath": str(logs / "service.err.log"),
        "EnvironmentVariables": {"HOME": str(_home(home))},
    }


def install(
    palace_path: str,
    sessions_root: str,
    *,
    interval: float = 15.0,
    home: Optional[str] = None,
    executable: Optional[str] = None,
    load: bool = True,
) -> dict:
    if not is_supported():
        raise ServiceError(f"no supported service manager for platform {sys.platform!r}")

    path = plist_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_dir(palace_path).mkdir(parents=True, exist_ok=True)

    definition = build_plist(
        palace_path, sessions_root, interval=interval, executable=executable, home=home
    )
    with open(path, "wb") as fh:
        plistlib.dump(definition, fh)

    loaded = False
    if load and shutil.which("launchctl"):
        _launchctl("bootout", f"{_domain()}/{LABEL}", home=home)
        result = _launchctl("bootstrap", _domain(), str(path), home=home)
        loaded = result.returncode == 0
    return {"plist": str(path), "label": LABEL, "loaded": loaded}


def start(home: Optional[str] = None) -> bool:
    if not is_supported() or not shutil.which("launchctl"):
        return False
    path = plist_path(home)
    if not path.exists():
        raise ServiceError(f"service is not installed ({path} missing)")
    _launchctl("bootstrap", _domain(), str(path), home=home)
    return _launchctl("kickstart", f"{_domain()}/{LABEL}", home=home).returncode == 0


def stop(home: Optional[str] = None) -> bool:
    if not is_supported() or not shutil.which("launchctl"):
        return False
    return _launchctl("bootout", f"{_domain()}/{LABEL}", home=home).returncode == 0


def is_installed(home: Optional[str] = None) -> bool:
    return plist_path(home).exists()


def is_loaded(home: Optional[str] = None) -> bool:
    if not is_supported() or not shutil.which("launchctl"):
        return False
    return _launchctl("print", f"{_domain()}/{LABEL}", home=home).returncode == 0


def uninstall(palace_path: str, *, home: Optional[str] = None) -> dict:
    """Remove only MemPalace-managed service files.

    The plist and this service's own log files are removed. The palace, its
    drawers, the follower heartbeat and anything the user authored are left
    exactly as they were — uninstalling capture must never look like data
    loss.
    """
    removed: list[str] = []
    path = plist_path(home)

    if is_supported() and shutil.which("launchctl"):
        _launchctl("bootout", f"{_domain()}/{LABEL}", home=home)

    if path.exists():
        path.unlink()
        removed.append(str(path))

    for name in ("service.out.log", "service.err.log"):
        log = log_dir(palace_path) / name
        if log.exists():
            log.unlink()
            removed.append(str(log))

    return {"removed": removed, "label": LABEL}
