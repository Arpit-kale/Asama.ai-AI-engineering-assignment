"""Pure functions: raw command/file text -> normalised Python structures.

No I/O here, so every parser is unit-tested against captured real-world output.
"""
from __future__ import annotations

import json
import re
import shlex
from typing import Any

_UNITS = {"b": 1, "kb": 1024, "kib": 1024, "mb": 1024**2, "mib": 1024**2,
          "gb": 1024**3, "gib": 1024**3, "tb": 1024**4, "tib": 1024**4}


# --------------------------------------------------------------------------- units
def parse_size(text: str | None) -> int | None:
    """'16 GB' / '8192 MB' / '512 kB' -> bytes (binary multiples, as dmidecode/meminfo use)."""
    if not text:
        return None
    m = re.match(r"^\s*([\d.]+)\s*([kmgt]i?b|b)?\s*$", text.strip(), re.IGNORECASE)
    if not m:
        return None
    return int(float(m.group(1)) * _UNITS[(m.group(2) or "b").lower()])


def human_bytes(n: int | float | None, binary: bool = True) -> str | None:
    """Binary (RAM): 17179869184 -> '16 GB'. Decimal (disks): 512110190592 -> '512.1 GB'."""
    if n is None:
        return None
    base = 1024 if binary else 1000
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < base or unit == "PB":
            return f"{value:.0f} {unit}" if value == int(value) else f"{value:.1f} {unit}"
        value /= base
    return None


def _to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if v in ("1", 1, "true"):
        return True
    if v in ("0", 0, "false"):
        return False
    return None


def _to_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- CPU
def parse_lscpu_json(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}

    def walk(items: list[dict]) -> None:
        for item in items:
            key = str(item.get("field", "")).strip().rstrip(":")
            if key and item.get("data") is not None:
                out[key] = str(item["data"]).strip()
            walk(item.get("children", []))

    walk(json.loads(stdout).get("lscpu", []))
    return out


def parse_lscpu_text(stdout: str) -> dict[str, str]:
    out = {}
    for line in stdout.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def parse_cpuinfo(text: str) -> dict[str, str]:
    """Fallback when lscpu is missing: summarise /proc/cpuinfo."""
    procs = [b for b in text.split("\n\n") if b.strip()]
    first: dict[str, str] = {}
    for line in procs[0].splitlines() if procs else []:
        if ":" in line:
            k, v = line.split(":", 1)
            first[k.strip()] = v.strip()
    phys = {ln.split(":", 1)[1].strip() for b in procs for ln in b.splitlines() if ln.startswith("physical id")}
    out = {"CPU(s)": str(len(procs)), "Model name": first.get("model name", first.get("Model", "")),
           "Vendor ID": first.get("vendor_id", "")}
    if first.get("cpu cores"):
        out["Core(s) per socket"] = first["cpu cores"]
    out["Socket(s)"] = str(len(phys) or 1)
    if first.get("flags"):
        out["Flags"] = first["flags"]
    return {k: v for k, v in out.items() if v}


def normalise_cpu(raw: dict[str, str]) -> dict[str, Any]:
    def num(key: str) -> int | None:
        return _to_int(raw.get(key, "").split()[0]) if raw.get(key) else None

    def mhz(key: str) -> float | None:
        try:
            return round(float(raw[key].replace(",", ".")), 1)
        except (KeyError, ValueError):
            return None

    logical = num("CPU(s)")
    cores_per_socket, sockets = num("Core(s) per socket"), num("Socket(s)") or 1
    threads_per_core = num("Thread(s) per core")
    physical = cores_per_socket * sockets if cores_per_socket else None
    flags = set(raw.get("Flags", "").split())
    return {
        "model": raw.get("Model name") or None,
        "vendor": raw.get("Vendor ID") or None,
        "architecture": raw.get("Architecture") or None,
        "sockets": sockets,
        "physical_cores": physical,
        "logical_cpus": logical,
        "threads_per_core": threads_per_core or (logical // physical if logical and physical else None),
        "max_mhz": mhz("CPU max MHz"),
        "min_mhz": mhz("CPU min MHz"),
        "current_mhz": mhz("CPU MHz"),
        "l2_cache": raw.get("L2 cache"),
        "l3_cache": raw.get("L3 cache"),
        "virtualization": raw.get("Virtualization"),
        "hypervisor": raw.get("Hypervisor vendor"),
        "supports_avx2": "avx2" in flags if flags else None,
        "supports_avx512": any(f.startswith("avx512") for f in flags) if flags else None,
    }


# --------------------------------------------------------------------------- memory
def parse_free_b(stdout: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == "Mem:" and len(parts) >= 7:
            out.update(total=int(parts[1]), used=int(parts[2]), free=int(parts[3]),
                       shared=int(parts[4]), buff_cache=int(parts[5]), available=int(parts[6]))
        elif parts and parts[0] == "Swap:" and len(parts) >= 4:
            out.update(swap_total=int(parts[1]), swap_used=int(parts[2]))
    return out


def parse_meminfo(text: str) -> dict[str, int]:
    vals = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s+(\d+)\s*(kB)?", line)
        if m:
            vals[m.group(1)] = int(m.group(2)) * (1024 if m.group(3) else 1)
    total, avail = vals.get("MemTotal"), vals.get("MemAvailable")
    out = {"total": total, "available": avail, "free": vals.get("MemFree"),
           "used": total - avail if total and avail is not None else None,
           "swap_total": vals.get("SwapTotal"),
           "swap_used": vals["SwapTotal"] - vals["SwapFree"] if "SwapTotal" in vals and "SwapFree" in vals else None}
    return {k: v for k, v in out.items() if v is not None}


def _dmi_blocks(stdout: str) -> list[tuple[str, dict[str, str]]]:
    """Split dmidecode output into (section title, {key: value}) blocks."""
    blocks = []
    for chunk in re.split(r"\n\s*\n", stdout):
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        if not lines or not lines[0].startswith("Handle"):
            continue
        title = lines[1].strip() if len(lines) > 1 else ""
        fields: dict[str, str] = {}
        for ln in lines[2:]:
            if ":" in ln and ln.startswith("\t") and not ln.startswith("\t\t"):
                k, v = ln.strip().split(":", 1)
                fields[k.strip()] = v.strip()
        blocks.append((title, fields))
    return blocks


_EMPTY_SLOT = ("no module installed", "not installed", "unknown", "0", "0 mb", "0 gb")


def parse_dmidecode_memory(stdout: str) -> dict[str, Any]:
    max_capacity, declared_slots, slots = 0, 0, []
    for title, f in _dmi_blocks(stdout):
        if title == "Physical Memory Array" and f.get("Use", "System Memory") == "System Memory":
            max_capacity += parse_size(f.get("Maximum Capacity")) or 0
            declared_slots += _to_int(f.get("Number Of Devices")) or 0
        elif title == "Memory Device":
            size_txt = f.get("Size", "")
            installed = size_txt.lower() not in _EMPTY_SLOT and parse_size(size_txt) is not None
            form = f.get("Form Factor")
            slots.append({
                "locator": f.get("Locator"),
                "bank": f.get("Bank Locator"),
                "installed": installed,
                "size_bytes": parse_size(size_txt) if installed else 0,
                "type": f.get("Type") if installed else None,
                "form_factor": form,
                "speed": f.get("Speed") if installed else None,
                "configured_speed": f.get("Configured Memory Speed") or f.get("Configured Clock Speed"),
                "manufacturer": f.get("Manufacturer") if installed else None,
                "part_number": f.get("Part Number") if installed else None,
                # Soldered memory reports "Row Of Chips" (or similar) instead of DIMM/SODIMM.
                "soldered": bool(form) and form.lower() in ("row of chips", "chip") if installed else None,
            })
    installed_total = sum(s["size_bytes"] for s in slots)
    used = sum(1 for s in slots if s["installed"])
    total_slots = len(slots) or declared_slots
    return {
        "max_capacity_bytes": max_capacity or None,
        "total_slots": total_slots,
        "used_slots": used,
        "free_slots": total_slots - used,
        "installed_bytes": installed_total,
        "slots": slots,
    }


def upgrade_analysis(topology: dict[str, Any]) -> dict[str, Any]:
    """Pre-computed upgrade facts so the LLM never has to do the arithmetic itself."""
    max_cap, installed = topology.get("max_capacity_bytes"), topology.get("installed_bytes") or 0
    total, free = topology.get("total_slots") or 0, topology.get("free_slots") or 0
    replaceable = [s for s in topology.get("slots", []) if s["installed"] and not s.get("soldered")]
    soldered_total = sum(s["size_bytes"] for s in topology.get("slots", []) if s.get("soldered"))
    out: dict[str, Any] = {
        "can_add_without_removing": free > 0,
        "headroom_bytes": max(0, max_cap - installed) if max_cap else None,
        "max_per_slot_bytes": (max_cap // total) if max_cap and total else None,
        "soldered_bytes": soldered_total,
        "replaceable_modules": len(replaceable),
    }
    if max_cap and free:
        # Largest stick you can add to the empty slot(s), bounded by board max and per-slot max.
        per_slot = out["max_per_slot_bytes"] or max_cap
        out["max_addable_now_bytes"] = min(out["headroom_bytes"], per_slot * free)
    else:
        out["max_addable_now_bytes"] = 0 if max_cap else None
    if max_cap:
        out["upgrade_possible"] = installed < max_cap and (free > 0 or bool(replaceable))
    else:
        out["upgrade_possible"] = None
    return out


# --------------------------------------------------------------------------- storage
def parse_lsblk_json(stdout: str) -> list[dict[str, Any]]:
    disks = []

    def mounts(dev: dict) -> list[str]:
        own = [dev["mountpoint"]] if dev.get("mountpoint") else []
        return own + [m for ch in dev.get("children", []) or [] for m in mounts(ch)]

    for dev in json.loads(stdout).get("blockdevices", []):
        if dev.get("type") != "disk":
            continue
        name, size = dev.get("name", ""), _to_int(dev.get("size")) or 0
        if size == 0 or name.startswith(("zram", "loop", "ram")):
            continue
        rota, tran = _to_bool(dev.get("rota")), (dev.get("tran") or "").lower() or None
        if tran == "nvme" or name.startswith("nvme"):
            kind = "NVMe SSD"
        elif tran in ("virtio", "xen") or name.startswith(("vd", "xvd")):
            kind = "virtual disk"
        elif rota is False:
            kind = "SATA/other SSD" if tran == "sata" else ("USB flash/SSD" if tran == "usb" else "SSD")
        elif rota is True:
            kind = "HDD (rotational)"
        else:
            kind = "unknown"
        disks.append({
            "name": name, "model": (dev.get("model") or "").strip() or None,
            "vendor": (dev.get("vendor") or "").strip() or None,
            "size_bytes": size, "size_human": human_bytes(size, binary=False),
            "transport": tran, "rotational": rota, "removable": _to_bool(dev.get("rm")),
            "kind": kind, "is_nvme": kind == "NVMe SSD", "mountpoints": mounts(dev),
        })
    return disks


# --------------------------------------------------------------------------- PCI / GPU
_ID_RE = re.compile(r"^(.*?)\s*\[([0-9a-fA-F]{4})\]$")


def _split_id(text: str) -> tuple[str, str | None]:
    m = _ID_RE.match(text.strip())
    return (m.group(1).strip(), m.group(2).lower()) if m else (text.strip(), None)


def parse_lspci_mm(stdout: str) -> list[dict[str, Any]]:
    devices = []
    for line in stdout.splitlines():
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        positional = [p for p in parts if not re.match(r"^-[rp][0-9a-fA-F]+$", p)]
        if len(positional) < 4:
            continue
        cls, cls_id = _split_id(positional[1])
        vendor, vendor_id = _split_id(positional[2])
        device, device_id = _split_id(positional[3])
        devices.append({"slot": positional[0], "class": cls, "class_id": cls_id, "vendor": vendor,
                        "vendor_id": vendor_id, "device": device, "device_id": device_id})
    return devices


GPU_CLASS_IDS = {"0300", "0301", "0302", "0380"}
PCI_VENDORS = {"8086": "Intel", "10de": "NVIDIA", "1002": "AMD", "1af4": "Red Hat (virtio)",
               "1234": "QEMU", "15ad": "VMware", "80ee": "VirtualBox", "1414": "Microsoft"}


def gpus_from_lspci(devices: list[dict]) -> list[dict]:
    return [d for d in devices if d["class_id"] in GPU_CLASS_IDS or
            any(w in d["class"].lower() for w in ("vga", "3d controller", "display controller"))]


def parse_nvidia_smi(stdout: str) -> list[dict[str, Any]]:
    out = []
    for line in stdout.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 6:
            continue
        mem_total = _to_int(cols[1])
        out.append({"name": cols[0], "vram_total_bytes": mem_total * 1024**2 if mem_total else None,
                    "vram_used_bytes": (_to_int(cols[2]) or 0) * 1024**2, "driver_version": cols[3],
                    "temperature_c": _to_int(cols[4]), "utilization_pct": _to_int(cols[5])})
    return out


# --------------------------------------------------------------------------- OS
def parse_os_release(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
    return out
