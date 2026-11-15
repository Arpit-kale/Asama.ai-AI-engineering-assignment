"""Hardware inspection tools exposed to the LLM.

Each tool:
* has a strict JSON-schema for its (few, scalar) parameters, validated before running;
* gathers data only through ``SystemAccess`` (allow-listed commands / files);
* tries the best source first and falls back to unprivileged ones (portability);
* returns ``{"data", "sources", "unavailable"}`` so the model can see exactly what is
  known, where it came from, and what could NOT be determined (and why).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import parsers as P
from .executor import ExecError, PermissionNeeded, SystemAccess


@dataclass
class Result:
    data: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    unavailable: list[dict[str, str]] = field(default_factory=list)

    def missing(self, what: str, reason: str) -> None:
        self.unavailable.append({"field": what, "reason": reason})

    def as_dict(self) -> dict[str, Any]:
        return {"data": self.data, "sources": self.sources, "unavailable": self.unavailable}


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Result]
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    ttl: float = 3600.0                     # static facts are cached; live metrics only briefly
    available: Callable[[SystemAccess], bool] = lambda s: True

    def spec(self) -> dict[str, Any]:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolArgumentError(ValueError):
    pass


def validate_args(schema: dict[str, Any], args: dict[str, Any] | None) -> dict[str, Any]:
    args = args or {}
    if not isinstance(args, dict):
        raise ToolArgumentError("arguments must be an object")
    props = schema.get("properties", {})
    unknown = set(args) - set(props)
    if unknown:
        raise ToolArgumentError(f"unknown argument(s): {', '.join(sorted(unknown))}")
    for req in schema.get("required", []):
        if req not in args:
            raise ToolArgumentError(f"missing required argument: {req}")
    clean = {}
    for key, value in args.items():
        rule = props[key]
        if value is None:
            continue
        if rule["type"] == "boolean":
            if not isinstance(value, bool):
                if str(value).lower() in ("true", "false"):
                    value = str(value).lower() == "true"
                else:
                    raise ToolArgumentError(f"{key} must be a boolean")
        elif rule["type"] == "string":
            if not isinstance(value, str) or len(value) > rule.get("maxLength", 64):
                raise ToolArgumentError(f"{key} must be a short string")
            if "pattern" in rule and not re.fullmatch(rule["pattern"], value):
                raise ToolArgumentError(f"{key} has an invalid format")
            if "enum" in rule and value not in rule["enum"]:
                raise ToolArgumentError(f"{key} must be one of {rule['enum']}")
        clean[key] = value
    return clean


def _try_run(sys: SystemAccess, cmd: str, res: Result) -> str | None:
    if not sys.available(cmd):
        return None
    try:
        out = sys.run(cmd)
        res.sources.append(f"command:{cmd}")
        return out
    except ExecError:
        return None


def _read(sys: SystemAccess, path: str, res: Result) -> str | None:
    text = sys.read_text(path)
    if text is not None and path not in res.sources:
        res.sources.append(path)
    return text


def _sysfs(sys: SystemAccess, path: str) -> str | None:
    text = sys.read_text(path)
    return text or None


# =========================================================================== tools
CHASSIS = {"3": "desktop", "4": "desktop", "6": "desktop", "7": "desktop", "8": "laptop", "9": "laptop",
           "10": "laptop", "11": "handheld", "13": "all-in-one", "14": "laptop", "30": "tablet",
           "31": "convertible laptop", "32": "detachable laptop", "17": "server", "23": "server", "1": "other"}


def system_overview(sys: SystemAccess) -> Result:
    res = Result()
    osr = _read(sys, "/etc/os-release", res) or _read(sys, "/usr/lib/os-release", res)
    if osr:
        info = P.parse_os_release(osr)
        res.data["os"] = info.get("PRETTY_NAME") or info.get("NAME")
    u = os.uname()
    res.data["kernel"] = f"{u.sysname} {u.release}"
    res.data["architecture"] = u.machine
    dmi = {}
    for key in ("sys_vendor", "product_name", "product_version", "product_family", "board_vendor",
                "board_name", "bios_vendor", "bios_version", "bios_date", "chassis_type"):
        val = _sysfs(sys, f"/sys/class/dmi/id/{key}")
        if val and val.lower() not in ("to be filled by o.e.m.", "default string", "not applicable"):
            dmi[key] = val
    if dmi:
        res.sources.append("/sys/class/dmi/id")
        res.data["manufacturer"] = dmi.get("sys_vendor")
        # Lenovo stores the marketing name (e.g. "ThinkPad X1 Carbon Gen 9") in product_version.
        lenovo = dmi.get("sys_vendor", "").upper() == "LENOVO" and dmi.get("product_version")
        res.data["model"] = dmi.get("product_version") if lenovo else dmi.get("product_name")
        res.data["product_code"] = dmi.get("product_name") if lenovo else None
        res.data["motherboard"] = " ".join(x for x in (dmi.get("board_vendor"), dmi.get("board_name")) if x) or None
        res.data["bios"] = " ".join(x for x in (dmi.get("bios_vendor"), dmi.get("bios_version"), dmi.get("bios_date")) if x) or None
        res.data["form_factor"] = CHASSIS.get(dmi.get("chassis_type", ""), "unknown")
    else:
        res.missing("manufacturer/model/bios", "/sys/class/dmi/id not present (VM, container or non-x86 board)")
    cpuinfo = _sysfs(sys, "/proc/cpuinfo") or ""
    vendor = (res.data.get("manufacturer") or "").lower()
    res.data["virtual_machine"] = bool(re.search(r"\bhypervisor\b", cpuinfo)) or any(
        v in vendor for v in ("qemu", "vmware", "innotek", "virtualbox", "microsoft corporation", "xen", "kvm"))
    return res


def cpu_info(sys: SystemAccess) -> Result:
    res = Result()
    raw = None
    out = _try_run(sys, "lscpu_json", res)
    if out:
        try:
            raw = P.parse_lscpu_json(out)
        except ValueError:
            raw = None
    if raw is None and (out := _try_run(sys, "lscpu", res)):
        raw = P.parse_lscpu_text(out)
    if raw is None and (text := _read(sys, "/proc/cpuinfo", res)):
        raw = P.parse_cpuinfo(text)
    if raw is None:
        res.missing("cpu", "neither lscpu nor /proc/cpuinfo is available")
        return res
    res.data = {k: v for k, v in P.normalise_cpu(raw).items() if v is not None}
    if "max_mhz" not in res.data:
        res.missing("max_mhz", "CPU does not report a maximum frequency (common in VMs)")
    return res


def memory_info(sys: SystemAccess) -> Result:
    res = Result()
    out = _try_run(sys, "free", res)
    mem = P.parse_free_b(out) if out else {}
    if not mem and (text := _read(sys, "/proc/meminfo", res)):
        mem = P.parse_meminfo(text)
    if not mem:
        res.missing("memory", "neither free nor /proc/meminfo is available")
        return res
    res.data = {f"{k}_bytes": v for k, v in mem.items()}
    res.data.update({f"{k}_human": P.human_bytes(v) for k, v in mem.items() if k in ("total", "used", "available", "swap_total")})
    res.data["note"] = ("total is the usable RAM reported by the kernel; it is slightly lower than the installed "
                        "physical RAM because firmware/integrated graphics reserve some.")
    return res


def memory_slots(sys: SystemAccess, include_empty_slots: bool = True) -> Result:
    res = Result()
    reason = None
    if not sys.available("dmidecode_memory"):
        reason = "dmidecode is not installed (install the 'dmidecode' package)"
    else:
        try:
            out = sys.run("dmidecode_memory")
            res.sources.append("command:dmidecode -t memory (root)")
            topo = P.parse_dmidecode_memory(out)
            if not topo["slots"] and not topo["max_capacity_bytes"]:
                reason = "firmware exposes no SMBIOS memory tables (typical inside VMs)"
            else:
                up = P.upgrade_analysis(topo)
                slots = topo["slots"] if include_empty_slots else [s for s in topo["slots"] if s["installed"]]
                for s in slots:
                    s["size_human"] = P.human_bytes(s["size_bytes"]) if s["installed"] else None
                res.data = {
                    "installed_total_human": P.human_bytes(topo["installed_bytes"]),
                    "installed_total_bytes": topo["installed_bytes"],
                    "total_slots": topo["total_slots"], "used_slots": topo["used_slots"],
                    "free_slots": topo["free_slots"],
                    "max_supported_human": P.human_bytes(topo["max_capacity_bytes"]),
                    "max_supported_bytes": topo["max_capacity_bytes"],
                    "slots": slots,
                    "upgrade": {**up, **{k.replace("_bytes", "_human"): P.human_bytes(v)
                                         for k, v in up.items() if k.endswith("_bytes") and v is not None}},
                }
                if not topo["max_capacity_bytes"]:
                    res.missing("max_supported", "firmware does not report a maximum capacity")
                return res
        except PermissionNeeded:
            reason = ("needs root (dmidecode). Run `hwqa --setup-sudo` for a scoped, password-less sudo rule, "
                      "or start with `--sudo prompt`.")
        except ExecError as exc:
            msg = str(exc)
            reason = ("firmware exposes no SMBIOS memory tables (typical inside VMs/containers)"
                      if "/dev/mem" in msg or "No SMBIOS" in msg else msg)
    res.missing("slots/max_supported/module_details", reason or "unknown error")
    # Partial answer from unprivileged sources.
    text = _read(sys, "/proc/meminfo", res)
    if text:
        total = P.parse_meminfo(text).get("total")
        res.data["usable_total_human"] = P.human_bytes(total)
        res.data["usable_total_bytes"] = total
    return res


def storage_devices(sys: SystemAccess, device: str | None = None) -> Result:
    res = Result()
    disks: list[dict] = []
    out = _try_run(sys, "lsblk", res)
    if out:
        try:
            disks = P.parse_lsblk_json(out)
        except ValueError:
            disks = []
    if not disks:                               # sysfs fallback
        for name in sys.list_dir("/sys/block"):
            if name.startswith(("loop", "ram", "zram", "dm-", "md", "sr")):
                continue
            size = int(_sysfs(sys, f"/sys/block/{name}/size") or 0) * 512
            if not size:
                continue
            rota = _sysfs(sys, f"/sys/block/{name}/queue/rotational")
            fake = {"name": name, "size": size, "type": "disk", "rota": rota,
                    "model": _sysfs(sys, f"/sys/block/{name}/device/model"),
                    "tran": "nvme" if name.startswith("nvme") else None,
                    "rm": _sysfs(sys, f"/sys/block/{name}/removable")}
            disks += P.parse_lsblk_json(json.dumps({"blockdevices": [fake]}))
        if disks:
            res.sources.append("/sys/block")
    if device:
        disks = [d for d in disks if d["name"] == device]
        if not disks:
            res.missing(device, "no such disk")
    if not disks and not device:
        res.missing("disks", "no physical disks detected")
    res.data = {"disks": disks, "count": len(disks),
                "any_nvme": any(d["is_nvme"] for d in disks)}
    return res


REAL_FS = {"ext2", "ext3", "ext4", "xfs", "btrfs", "vfat", "exfat", "ntfs", "ntfs3", "f2fs", "zfs", "fuseblk", "overlay"}


def disk_usage(sys: SystemAccess) -> Result:
    res = Result()
    mounts = _read(sys, "/proc/mounts", res)
    if not mounts:
        res.missing("filesystems", "/proc/mounts not readable")
        return res
    seen, fs = set(), []
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[2] not in REAL_FS or parts[0] in seen:
            continue
        mnt = parts[1].replace("\\040", " ")
        if mnt.startswith(("/snap/", "/proc", "/sys", "/run/credentials")):
            continue
        seen.add(parts[0])
        try:
            st = os.statvfs(mnt)
        except OSError:
            continue
        total, free = st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize
        if not total:
            continue
        fs.append({"mountpoint": mnt, "fstype": parts[2], "device": parts[0],
                   "total_human": P.human_bytes(total, binary=False), "free_human": P.human_bytes(free, binary=False),
                   "used_pct": round(100 * (total - free) / total, 1), "total_bytes": total, "free_bytes": free})
    res.data = {"filesystems": fs}
    return res


def gpu_info(sys: SystemAccess) -> Result:
    res = Result()
    gpus: list[dict] = []
    out = _try_run(sys, "lspci", res)
    if out:
        for d in P.gpus_from_lspci(P.parse_lspci_mm(out)):
            gpus.append({"vendor": d["vendor"], "model": d["device"], "pci_slot": d["slot"],
                         "type": "discrete" if d["class_id"] == "0302" or d["vendor_id"] in ("10de",) else
                                 ("integrated" if d["vendor_id"] == "8086" else "unknown")})
    else:                                        # sysfs fallback: PCI IDs only
        for card in sys.list_dir("/sys/class/drm"):
            if not re.fullmatch(r"card\d+", card):
                continue
            vid = (_sysfs(sys, f"/sys/class/drm/{card}/device/vendor") or "").replace("0x", "")
            did = (_sysfs(sys, f"/sys/class/drm/{card}/device/device") or "").replace("0x", "")
            if vid:
                gpus.append({"vendor": P.PCI_VENDORS.get(vid, f"vendor id {vid}"), "model": f"PCI device id {did}",
                             "driver": sys.link_name(f"/sys/class/drm/{card}/device/driver")})
        if gpus:
            res.sources.append("/sys/class/drm")
            res.missing("gpu_model_name", "lspci (pciutils) not installed; only PCI IDs are known")
    if sys.available("nvidia_smi") and (out := _try_run(sys, "nvidia_smi", res)):
        for n in P.parse_nvidia_smi(out):
            n["vram_total_human"] = P.human_bytes(n["vram_total_bytes"])
            res.data.setdefault("nvidia", []).append(n)
    elif any(g["vendor"].upper().startswith("NVIDIA") for g in gpus):
        res.missing("vram/driver", "nvidia-smi not available (proprietary driver not installed)")
    if not gpus and not res.data.get("nvidia"):
        res.missing("gpu", "no display controller found (headless server or VM without virtual GPU)"
                    if res.sources else "lspci is not installed and /sys/class/drm is empty")
    res.data["gpus"] = gpus
    return res


def network_interfaces(sys: SystemAccess) -> Result:
    res = Result()
    ifaces = []
    for name in sys.list_dir("/sys/class/net"):
        if name == "lo":
            continue
        base = f"/sys/class/net/{name}"
        entries = sys.list_dir(base)
        is_physical = "device" in entries
        speed = _sysfs(sys, f"{base}/speed")
        ifaces.append({
            "name": name,
            "kind": "wifi" if "wireless" in entries or "phy80211" in entries else ("ethernet" if is_physical else "virtual"),
            "state": _sysfs(sys, f"{base}/operstate"),
            "link_speed_mbps": int(speed) if speed and speed.lstrip("-").isdigit() and int(speed) > 0 else None,
            "driver": sys.link_name(f"{base}/device/driver") if is_physical else None,
        })
    if ifaces:
        res.sources.append("/sys/class/net")
    out = _try_run(sys, "lspci", res)
    if out:
        res.data["controllers"] = [f"{d['vendor']} {d['device']}" for d in P.parse_lspci_mm(out)
                                   if d["class_id"] in ("0200", "0280")]
    res.data["interfaces"] = ifaces
    return res


def battery_info(sys: SystemAccess) -> Result:
    res = Result()
    batteries, ac_online = [], None
    for name in sys.list_dir("/sys/class/power_supply"):
        base = f"/sys/class/power_supply/{name}"
        kind = _sysfs(sys, f"{base}/type")
        if kind == "Mains":
            ac_online = _sysfs(sys, f"{base}/online") == "1"
        if kind != "Battery":
            continue
        full = _sysfs(sys, f"{base}/energy_full") or _sysfs(sys, f"{base}/charge_full")
        design = _sysfs(sys, f"{base}/energy_full_design") or _sysfs(sys, f"{base}/charge_full_design")
        health = round(100 * int(full) / int(design), 1) if full and design and int(design) else None
        cycles = _sysfs(sys, f"{base}/cycle_count")
        batteries.append({
            "name": name, "status": _sysfs(sys, f"{base}/status"),
            "charge_pct": int(c) if (c := _sysfs(sys, f"{base}/capacity")) and c.isdigit() else None,
            "health_pct": health,
            "cycle_count": int(cycles) if cycles and cycles.isdigit() and int(cycles) > 0 else None,
            "technology": _sysfs(sys, f"{base}/technology"),
            "manufacturer": _sysfs(sys, f"{base}/manufacturer"), "model": _sysfs(sys, f"{base}/model_name"),
        })
    res.sources.append("/sys/class/power_supply")
    res.data = {"batteries": batteries, "ac_adapter_online": ac_online, "has_battery": bool(batteries)}
    return res


def temperatures(sys: SystemAccess) -> Result:
    res = Result()
    sensors = []
    for zone in sys.list_dir("/sys/class/thermal"):
        if zone.startswith("thermal_zone"):
            t = _sysfs(sys, f"/sys/class/thermal/{zone}/temp")
            if t and t.lstrip("-").isdigit():
                sensors.append({"sensor": _sysfs(sys, f"/sys/class/thermal/{zone}/type") or zone,
                                "celsius": round(int(t) / 1000, 1)})
    for hw in sys.list_dir("/sys/class/hwmon"):
        base = f"/sys/class/hwmon/{hw}"
        chip = _sysfs(sys, f"{base}/name") or hw
        for f in sys.list_dir(base):
            if re.fullmatch(r"temp\d+_input", f):
                t = _sysfs(sys, f"{base}/{f}")
                label = _sysfs(sys, f"{base}/{f.replace('_input', '_label')}")
                if t and t.lstrip("-").isdigit():
                    sensors.append({"sensor": f"{chip} {label or f.split('_')[0]}", "celsius": round(int(t) / 1000, 1)})
    if sensors:
        res.sources.append("/sys/class/thermal,/sys/class/hwmon")
    else:
        res.missing("temperatures", "no thermal sensors exposed (common in VMs)")
    res.data = {"sensors": sensors}
    return res


# =========================================================================== registry
def build_tools() -> list[Tool]:
    return [
        Tool("get_system_overview", "Laptop manufacturer and model, form factor, motherboard, BIOS, OS/distro, "
             "kernel and whether this is a virtual machine.", system_overview),
        Tool("get_cpu_info", "CPU model, vendor, physical cores, logical threads, clock speeds (MHz), caches, "
             "virtualization and instruction-set support.", cpu_info),
        Tool("get_memory_info", "Current RAM usage: usable total, used, available, and swap. Use for "
             "'how much RAM do I have' or 'how much memory is free'.", memory_info, ttl=5),
        Tool("get_memory_slots", "Physical RAM topology from firmware: installed modules per slot (size, type "
             "e.g. DDR4, speed), total/used/free slots, maximum supported capacity and a pre-computed upgrade "
             "analysis. Use for any RAM upgrade question.", memory_slots,
             {"type": "object", "properties": {"include_empty_slots": {
                 "type": "boolean", "description": "Include unused slots in the list (default true)."}}}),
        Tool("get_storage_devices", "Physical disks: model, size, NVMe/SSD/HDD type, transport (nvme/sata/usb) "
             "and mount points.", storage_devices,
             {"type": "object", "properties": {"device": {
                 "type": "string", "pattern": r"[a-zA-Z0-9]{1,32}", "maxLength": 32,
                 "description": "Optional kernel disk name to filter on, e.g. 'nvme0n1' or 'sda'."}}}),
        Tool("get_disk_usage", "Free and used space per mounted filesystem.", disk_usage, ttl=5),
        Tool("get_gpu_info", "Graphics adapters (integrated/discrete), vendor and model; NVIDIA VRAM, driver "
             "and temperature when nvidia-smi exists.", gpu_info),
        Tool("get_network_interfaces", "Network adapters: Wi-Fi/Ethernet, link state, speed, driver and "
             "controller model.", network_interfaces, ttl=10),
        Tool("get_battery_info", "Battery presence, charge %, health % (full vs design capacity), cycle count "
             "and whether AC power is connected.", battery_info, ttl=10),
        Tool("get_temperatures", "Current temperature readings from thermal sensors (CPU, etc.) in Celsius.",
             temperatures, ttl=3),
    ]


class ToolRegistry:
    def __init__(self, system: SystemAccess) -> None:
        self.system = system
        self.tools = {t.name: t for t in build_tools() if t.available(system)}
        self._cache: dict[tuple, tuple[float, dict]] = {}

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def call(self, name: str, args: dict | None = None) -> dict[str, Any]:
        """Never raises: errors are returned as data so the LLM can recover or say 'I don't know'."""
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool '{name}'. Available: {', '.join(self.tools)}"}
        try:
            clean = validate_args(tool.parameters, args)
        except ToolArgumentError as exc:
            return {"error": f"invalid arguments for {name}: {exc}"}
        key = (name, tuple(sorted(clean.items())))
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < tool.ttl:
            return hit[1]
        try:
            result = tool.func(self.system, **clean).as_dict()
        except Exception as exc:  # noqa: BLE001 - a broken parser must not crash the chat
            result = {"error": f"{name} failed: {type(exc).__name__}: {exc}"}
        self._cache[key] = (time.monotonic(), result)
        return result

    def clear_cache(self) -> None:
        self._cache.clear()

