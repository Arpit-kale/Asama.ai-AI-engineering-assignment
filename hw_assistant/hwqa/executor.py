"""Safe system access: the ONLY place that touches the OS.

Security model
--------------
* The LLM never produces command strings. Tools refer to *command IDs* in ``COMMANDS``;
  each maps to a fixed binary and fixed argument vector. No LLM-supplied bytes reach execve.
* Binaries are resolved once, from trusted system directories only (not $PATH), to absolute paths.
* ``subprocess.run([...], shell=False)`` - no shell, so ``;``, ``&&``, ``|``, ``$()`` mean nothing.
* Minimal environment (``LC_ALL=C`` for stable parsing), timeout, output size cap, stdin closed.
* Root-only commands go through ``sudo -n <abs-path> <fixed args>``; a scoped sudoers rule can
  allow exactly those argument vectors and nothing else (see ``sudoers_rule``). The assistant
  itself never runs as root.
* File reads are restricted to /sys and a handful of named /proc and /etc files,
  after resolving symlinks, with a size cap.
"""
from __future__ import annotations

import getpass
import os
import subprocess
from dataclasses import dataclass
from enum import Enum

TRUSTED_DIRS = ("/usr/sbin", "/usr/bin", "/sbin", "/bin", "/usr/local/sbin", "/usr/local/bin")
SAFE_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C", "LANG": "C"}
MAX_OUTPUT_BYTES = 2_000_000
ALLOWED_READ_PREFIXES = ("/sys/",)
ALLOWED_READ_FILES = ("/etc/os-release", "/usr/lib/os-release", "/proc/cpuinfo", "/proc/meminfo",
                      "/proc/mounts", "/proc/version", "/proc/swaps")


@dataclass(frozen=True)
class CommandSpec:
    binary: str
    args: tuple[str, ...]
    needs_root: bool = False
    timeout: float = 5.0


# The complete allow-list. Read-only commands only.
COMMANDS: dict[str, CommandSpec] = {
    "lscpu_json": CommandSpec("lscpu", ("-J",)),
    "lscpu": CommandSpec("lscpu", ()),
    "free": CommandSpec("free", ("-b",)),
    "lsblk": CommandSpec("lsblk", ("-J", "-b", "-o", "NAME,SIZE,TYPE,MODEL,VENDOR,ROTA,TRAN,MOUNTPOINT,FSTYPE,RM")),
    "lspci": CommandSpec("lspci", ("-mm", "-nn")),
    "dmidecode_memory": CommandSpec("dmidecode", ("-t", "memory"), needs_root=True),
    "nvidia_smi": CommandSpec("nvidia-smi", (
        "--query-gpu=name,memory.total,memory.used,driver_version,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits")),
}


class SudoMode(str, Enum):
    NEVER = "never"                    # never escalate; use non-root fallbacks
    NONINTERACTIVE = "noninteractive"  # sudo -n: works only with a scoped NOPASSWD rule
    PROMPT = "prompt"                  # sudo may ask for the password once in the terminal


class ExecError(Exception):
    """Base class: the command could not produce output."""


class CommandUnavailable(ExecError):
    pass


class PermissionNeeded(ExecError):
    pass


class SystemAccess:
    """Interface used by tools (lets tests inject canned outputs)."""

    def available(self, command_id: str) -> bool: ...
    def run(self, command_id: str) -> str: ...
    def read_text(self, path: str) -> str | None: ...
    def list_dir(self, path: str) -> list[str]: ...
    def link_name(self, path: str) -> str | None: ...
    def is_root(self) -> bool: ...


def _resolve(binary: str) -> str | None:
    for d in TRUSTED_DIRS:
        path = os.path.join(d, binary)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return os.path.realpath(path)
    return None


class SafeExecutor(SystemAccess):
    def __init__(self, sudo_mode: SudoMode = SudoMode.NONINTERACTIVE) -> None:
        self.sudo_mode = sudo_mode
        self._paths = {cid: _resolve(spec.binary) for cid, spec in COMMANDS.items()}
        self._sudo = _resolve("sudo")

    # ---- discovery -----------------------------------------------------------
    def available(self, command_id: str) -> bool:
        return self._paths.get(command_id) is not None

    def capabilities(self) -> dict[str, str | None]:
        return dict(self._paths)

    def is_root(self) -> bool:
        return hasattr(os, "geteuid") and os.geteuid() == 0

    # ---- commands ------------------------------------------------------------
    def build_argv(self, command_id: str) -> list[str]:
        if command_id not in COMMANDS:
            raise PermissionError(f"command '{command_id}' is not allow-listed")
        spec, path = COMMANDS[command_id], self._paths.get(command_id)
        if not path:
            raise CommandUnavailable(f"'{spec.binary}' is not installed")
        argv = [path, *spec.args]
        if spec.needs_root and not self.is_root():
            if self.sudo_mode is SudoMode.NEVER or not self._sudo:
                raise PermissionNeeded(f"'{spec.binary}' needs root privileges")
            prefix = [self._sudo, "-n"] if self.sudo_mode is SudoMode.NONINTERACTIVE else [self._sudo]
            argv = prefix + argv
        return argv

    def run(self, command_id: str) -> str:
        spec = COMMANDS.get(command_id)
        argv = self.build_argv(command_id)
        interactive = argv[0] == self._sudo and self.sudo_mode is SudoMode.PROMPT
        try:
            proc = subprocess.run(
                argv, shell=False, capture_output=True, timeout=30 if interactive else spec.timeout,
                env=SAFE_ENV, stdin=None if interactive else subprocess.DEVNULL, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecError(f"'{spec.binary}' timed out") from exc
        except OSError as exc:
            raise ExecError(f"'{spec.binary}' failed to start: {exc}") from exc

        stdout = proc.stdout[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        stderr = proc.stderr[:4000].decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            low = stderr.lower()
            sudo_refused = argv[0] == self._sudo and (low.startswith("sudo:") or "password is required" in low)
            if sudo_refused or "permission denied" in low or "must be run as root" in low:
                raise PermissionNeeded(f"'{spec.binary}' needs root privileges ({stderr or 'sudo refused'})")
            raise ExecError(f"'{spec.binary}' exited with {proc.returncode}: {stderr}")
        return stdout

    # ---- files ---------------------------------------------------------------
    @staticmethod
    def _allowed_path(path: str) -> str | None:
        norm = os.path.normpath(path)
        real = os.path.realpath(norm)
        if real.startswith(ALLOWED_READ_PREFIXES) or real in ALLOWED_READ_FILES:
            return real
        # Named files that are symlinks by design (/proc/mounts -> /proc/<pid>/mounts,
        # /etc/os-release -> /usr/lib/os-release); the target must stay in /proc or be allow-listed.
        if norm in ALLOWED_READ_FILES and real.startswith("/proc/"):
            return real
        return None

    def read_text(self, path: str) -> str | None:
        real = self._allowed_path(path)
        if real is None:
            raise PermissionError(f"reading '{path}' is not allowed")
        try:
            with open(real, "rb") as fh:
                return fh.read(256_000).decode("utf-8", errors="replace").strip()
        except OSError:
            return None

    def list_dir(self, path: str) -> list[str]:
        real = self._allowed_path(path.rstrip("/") + "/")
        if real is None:
            raise PermissionError(f"listing '{path}' is not allowed")
        try:
            return sorted(os.listdir(real))
        except OSError:
            return []

    def link_name(self, path: str) -> str | None:
        """Basename of a sysfs symlink target (e.g. the driver bound to a device)."""
        if self._allowed_path(path) is None or not os.path.islink(path):
            return None
        return os.path.basename(os.path.realpath(path))


def sudoers_rule() -> str:
    """A least-privilege sudoers line covering exactly the root-only allow-listed commands."""
    user = getpass.getuser()
    entries = []
    for spec in COMMANDS.values():
        if spec.needs_root:
            path = _resolve(spec.binary) or f"/usr/sbin/{spec.binary}"
            entries.append(f"{path} {' '.join(spec.args)}")
    return f"{user} ALL=(root) NOPASSWD: " + ", ".join(entries)


def which_report() -> dict[str, bool]:
    return {spec.binary: _resolve(spec.binary) is not None for spec in COMMANDS.values()}

