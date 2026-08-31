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


def resolve_executable(executable: Optional[str] = None) -> tuple[str, dict]:
    """The exact interpreter + environment that can import *this* MemPalace.

    A launchd job that runs a bare ``mempalace`` off PATH is a coin flip: the
    login PATH is not the shell PATH, and on a machine with a venv, a uv tool
    install and a Homebrew python, it may resolve to a build that does not
    contain this feature at all — capture would "run" and quietly do the wrong
    thing. So the plist gets an absolute interpreter path, and installation
    refuses unless that interpreter imports the same package file this process
    is running from.

    Returns ``(executable, env)``. ``env`` carries ``PYTHONPATH`` when the
    package is only importable from its source tree (editable / worktree
    installs), so the job does not depend on site-packages layout.
    """
    import tempfile

    import mempalace

    candidate = executable or sys.executable
    here = str(Path(mempalace.__file__).resolve().parent)
    _PROBE = "import mempalace,sys; sys.stdout.write(mempalace.__file__)"

    def _probe(pythonpath: Optional[str]) -> str:
        # Run from a neutral directory: launched from the repo root, *any*
        # interpreter "imports mempalace" off the current directory, so a
        # cwd-inherited probe would happily bless a python that has no
        # MemPalace installed at all. PYTHONPATH is dropped for the same
        # reason unless we are the ones setting it.
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        if pythonpath:
            env["PYTHONPATH"] = pythonpath
        try:
            done = subprocess.run(
                [candidate, "-c", _PROBE],
                capture_output=True,
                text=True,
                check=False,
                cwd=tempfile.gettempdir(),
                env=env,
            )
        except OSError:
            return ""
        if done.returncode != 0 or not done.stdout.strip():
            return ""
        return str(Path(done.stdout.strip()).resolve().parent)

    env: dict[str, str] = {}
    resolved = _probe(None)
    if resolved != here:
        # The PYTHONPATH fallback exists for running out of a source tree or
        # worktree, where *this* interpreter already has the dependencies but
        # the package is not on its path. It is deliberately not offered to a
        # foreign interpreter: pointing some other python at this source tree
        # produces a service that starts and then dies on the first import of
        # a dependency it does not have.
        parent = str(Path(here).parent)
        resolved2 = _probe(parent) if candidate == sys.executable else ""
        if resolved2 != here:
            raise ServiceError(
                f"{candidate} does not import this MemPalace "
                f"(wanted {here}, got {resolved or resolved2 or 'nothing'}); "
                "install the service from the environment you want it to run"
            )
        env["PYTHONPATH"] = parent

    return candidate, env


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
    resolved_exe, extra_env = resolve_executable(executable)
    program = [
        resolved_exe,
        "-m",
        "mempalace",
        "codex-capture",
        "--palace",
        str(palace_path),
        "--sessions-root",
        str(sessions_root),
        "--interval",
        str(interval),
        # The service never touches the user's AGENTS.md. Recall is installed
        # only by an explicit `mempalace codex-recall install`.
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
        "EnvironmentVariables": {"HOME": str(_home(home)), **extra_env},
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
