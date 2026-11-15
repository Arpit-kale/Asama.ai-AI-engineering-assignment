"""Deterministic engine: works with no LLM at all.

1. Intent routing - keyword rules pick the tools to run (with follow-up memory).
2. Answer templates - every sentence is built from tool data; anything missing is reported
   as unknown with the reason. It cannot hallucinate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .tools import ToolRegistry

Fetch = Callable[[str], dict]


@dataclass(frozen=True)
class Intent:
    name: str
    pattern: re.Pattern
    tools: tuple[str, ...]


def _rx(*words: str) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(words) + r")\b", re.IGNORECASE)


INTENTS: list[Intent] = [
    Intent("ram_upgrade", re.compile(
        r"(upgrad\w*|add\w*|more|expand\w*|increase|extend|max(imum)?|support\w*|slots?|dimm|stick)"
        r".{0,40}\b(ram|memory)\b|\b(ram|memory)\b.{0,40}(upgrad\w*|slots?|max(imum)?|expand\w*|add\w*|support\w*)"
        r"|\b(memory|ram|dimm) slots?\b", re.IGNORECASE),
        ("get_memory_slots", "get_memory_info")),
    Intent("ram", _rx("ram", "memory", "ddr\\d?", "swap"), ("get_memory_info", "get_memory_slots")),
    Intent("disk_usage", re.compile(r"\b(space|free|full|usage|left|used)\b.{0,30}\b(disk|drive|storage|ssd|partition)s?\b"
                                    r"|\b(disk|drive|storage)\b.{0,30}\b(space|full|usage|left)\b|\bdisk space\b",
                                    re.IGNORECASE), ("get_disk_usage",)),
    Intent("storage", _rx("ssd", "nvme", "hdd", "disks?", "drives?", "storage", "hard ?drive", "sata"),
           ("get_storage_devices",)),
    Intent("cpu", _rx("cpu", "processor", "cores?", "threads?", "ghz", "clock", "intel", "amd", "ryzen", "xeon",
                      "core i\\d", "avx\\w*", "virtuali[sz]ation"), ("get_cpu_info",)),
    Intent("gpu", _rx("gpu", "graphics", "video card", "vga", "nvidia", "radeon", "geforce", "vram", "cuda",
                      "display adapter"), ("get_gpu_info",)),
    Intent("battery", _rx("battery", "batteries", "charge", "charging", "charger", "cycle count", "ac adapter"),
           ("get_battery_info",)),
    Intent("network", _rx("wifi", "wi-fi", "wireless", "ethernet", "network", "nic", "lan", "wlan", "internet adapter"),
           ("get_network_interfaces",)),
    Intent("temps", _rx("temperature", "temp", "temps", "hot", "thermal", "heat", "overheat\\w*"),
           ("get_temperatures",)),
    Intent("overview", _rx("model", "laptop", "manufacturer", "vendor", "make", "brand", "bios", "motherboard",
                           "os", "distro", "kernel", "linux", "specs?", "summary", "overview", "machine", "computer",
                           "everything", "virtual machine", "vm"), ("get_system_overview",)),
]

FOLLOW_UP = re.compile(r"^(and|what about|how about|also|is it|is that|does it|it|its|it's|that|which|how fast|how big|"
                       r"how much|how many|why|and how)\b", re.IGNORECASE)
PRONOUN = re.compile(r"\b(it|its|it's|that one|those|them|they|this one)\b", re.IGNORECASE)

HELP = ("I can only answer questions about this computer's hardware that I can read from the system: "
        "CPU, RAM (including upgrade options), disks/SSD type, free disk space, GPU, network adapters, "
        "battery, temperatures and the laptop model/OS. Try: \"How much RAM do I have?\", "
        "\"Is my SSD NVMe?\" or \"Can I add more RAM?\"")


def route(question: str, last: list[str]) -> list[str]:
    matched = [i.name for i in INTENTS if i.pattern.search(question)]
    if "ram_upgrade" in matched and "ram" in matched:
        matched.remove("ram")
    if "disk_usage" in matched and "storage" in matched and not re.search(r"nvme|ssd|hdd|type|model", question, re.I):
        matched.remove("storage")
    # "laptop"/"computer" are generic words: only keep overview if nothing more specific matched.
    if "overview" in matched and len(matched) > 1 and not re.search(r"model|manufacturer|vendor|bios|motherboard|os\b|distro|kernel|specs|summary|overview|everything", question, re.I):
        matched.remove("overview")
    if not matched and last and (FOLLOW_UP.match(question.strip()) or PRONOUN.search(question)):
        return last
    return matched


# =========================================================================== composers
def _unknown(result: dict) -> str:
    if "error" in result:
        return result["error"]
    return "; ".join(f"{u['field']}: {u['reason']}" for u in result.get("unavailable", []))


def _ghz(mhz: float | None) -> str | None:
    return f"{mhz / 1000:.2f} GHz" if mhz else None


def say_ram(fetch: Fetch) -> str:
    info, slots = fetch("get_memory_info"), fetch("get_memory_slots")
    d, s = info.get("data", {}), slots.get("data", {})
    parts = []
    if s.get("slots"):
        mods = [m for m in s["slots"] if m["installed"]]
        kinds = {m["type"] for m in mods if m.get("type")}
        speeds = {m["speed"] for m in mods if m.get("speed") and m["speed"].lower() != "unknown"}
        detail = ", ".join(filter(None, ["/".join(sorted(kinds)), "/".join(sorted(speeds))]))
        parts.append(f"You have {s['installed_total_human']} of RAM installed{f' ({detail})' if detail else ''}, "
                     f"using {s['used_slots']} of {s['total_slots']} slots.")
    if d.get("total_human"):
        lead = "The system reports" if parts else "Your system has"
        parts.append(f"{lead} {d['total_human']} usable, with {d['available_human']} available right now"
                     + (f" ({d['used_human']} in use)." if d.get("used_human") else "."))
        if d.get("swap_total_bytes"):
            parts.append(f"Swap: {d['swap_total_human']}.")
    if not s.get("slots"):
        parts.append(f"I can't see the physical memory modules or slots ({_unknown(slots)}).")
    return " ".join(parts) or f"I couldn't read memory information: {_unknown(info)}"


def say_ram_upgrade(fetch: Fetch) -> str:
    slots = fetch("get_memory_slots")
    s = slots.get("data", {})
    if not s.get("slots"):
        usable = s.get("usable_total_human")
        known = f" What I do know: {usable} of RAM is usable." if usable else ""
        return (f"I can't determine whether you can add RAM, because the slot and maximum-capacity data is "
                f"unavailable ({_unknown(slots)}).{known}")
    up = s["upgrade"]
    installed, used, total = s["installed_total_human"], s["used_slots"], s["total_slots"]
    mods = [m for m in s["slots"] if m["installed"]]
    sizes = " + ".join(m["size_human"] for m in mods)
    head = f"You have {installed} in {used} of {total} slot{'s' if total != 1 else ''}"
    head += f" ({sizes})." if used > 1 else "."
    if up.get("soldered_bytes"):
        head += f" {up.get('soldered_human', '')} of it is soldered to the board and can't be replaced."
    max_h = s.get("max_supported_human")
    if not max_h:
        tail = (f" There {'is' if s['free_slots'] == 1 else 'are'} {s['free_slots']} free slot(s), but the firmware "
                "doesn't report a maximum capacity, so I can't say how much the board supports. Check the "
                "manufacturer's specs for this model.")
        return head + tail
    if s["installed_total_bytes"] >= s["max_supported_bytes"]:
        return head + f" The board supports up to {max_h}, so you're already at the maximum."
    if s["free_slots"] > 0:
        per = up.get("max_per_slot_human")
        add = up.get("max_addable_now_human")
        n = s["free_slots"]
        if n == 1:
            stick = f"a stick of up to {add}"
        else:
            stick = f"up to {add} across the {n} free slots" + (f" (max {per} per slot)" if per else "")
        tail = f" The board supports up to {max_h}, so you can add {stick}."
        if mods and mods[0].get("type"):
            ff = f" {mods[0]['form_factor']}" if mods[0].get("form_factor") else ""
            tail += f" Use the same type as your current module ({mods[0]['type']}{ff})."
        return head + tail
    if up.get("replaceable_modules"):
        per = up.get("max_per_slot_human")
        return head + (f" All slots are full, but the board supports up to {max_h}, so you could replace the "
                       f"existing modules with larger ones (up to {per} per slot).")
    return head + f" The board lists {max_h} as its maximum, but there are no free or replaceable slots."


def say_cpu(fetch: Fetch) -> str:
    r = fetch("get_cpu_info")
    d = r.get("data", {})
    if not d.get("model") and not d.get("logical_cpus"):
        return f"I couldn't determine CPU details: {_unknown(r)}"
    s = f"Your CPU is {d.get('model', 'an unknown model')}"
    if d.get("physical_cores"):
        s += f" with {d['physical_cores']} physical core{'s' if d['physical_cores'] != 1 else ''} and {d.get('logical_cpus')} threads"
    elif d.get("logical_cpus"):
        s += f" with {d['logical_cpus']} logical CPUs"
    s += "."
    if d.get("max_mhz"):
        s += f" Max clock: {_ghz(d['max_mhz'])}" + (f" (min {_ghz(d['min_mhz'])})." if d.get("min_mhz") else ".")
    else:
        s += " The maximum clock speed isn't reported."
    if d.get("l3_cache"):
        s += f" L3 cache: {d['l3_cache']}."
    if d.get("hypervisor"):
        s += f" Note: this is a virtual CPU running under {d['hypervisor']}."
    return s


def say_storage(fetch: Fetch, question: str = "") -> str:
    r = fetch("get_storage_devices")
    disks = r.get("data", {}).get("disks", [])
    if not disks:
        return f"I couldn't find any physical disks ({_unknown(r) or 'none reported'})."
    lines = []
    for d in disks:
        desc = " ".join(filter(None, [d.get("vendor") if d.get("vendor") and not d["vendor"].startswith("0x") else None,
                                      d.get("model")])) or "unknown model"
        mounted = f", mounted at {', '.join(d['mountpoints'][:3])}" if d["mountpoints"] else ""
        lines.append(f"{d['name']}: {desc}, {d['size_human']}, {d['kind']}{mounted}")
    s = "Your disk" + ("s: " if len(disks) > 1 else ": ") + "; ".join(lines) + "."
    if re.search(r"nvme", question, re.I):
        nv = [d["name"] for d in disks if d["is_nvme"]]
        s = (f"Yes, {', '.join(nv)} {'is an' if len(nv) == 1 else 'are'} NVMe SSD{'s' if len(nv) > 1 else ''}. " if nv
             else "No, none of your disks are NVMe. ") + s
    elif re.search(r"\bssd\b|\bhdd\b", question, re.I):
        ssd = [d["name"] for d in disks if d["rotational"] is False]
        if any(d["kind"] == "virtual disk" for d in disks):
            s = "These are virtual disks, so their physical type can't be determined from inside the VM. " + s
        else:
            s = (f"{', '.join(ssd)} {'is' if len(ssd) == 1 else 'are'} solid-state. " if ssd
                 else "All your disks are rotational hard drives (HDD). ") + s
    return s


def say_disk_usage(fetch: Fetch) -> str:
    r = fetch("get_disk_usage")
    fs = r.get("data", {}).get("filesystems", [])
    if not fs:
        return f"I couldn't read filesystem usage ({_unknown(r)})."
    return "Disk space: " + "; ".join(
        f"{f['mountpoint']} has {f['free_human']} free of {f['total_human']} ({f['used_pct']:.0f}% used)" for f in fs) + "."


def say_gpu(fetch: Fetch) -> str:
    r = fetch("get_gpu_info")
    d = r.get("data", {})
    gpus, nvidia = d.get("gpus", []), d.get("nvidia", [])
    if not gpus and not nvidia:
        return f"I couldn't detect a GPU ({_unknown(r)})."
    parts = []
    for g in gpus:
        kind = f" ({g['type']})" if g.get("type") and g["type"] != "unknown" else ""
        parts.append(f"{g['vendor']} {g['model']}{kind}")
    s = ("Your GPU: " if len(parts) == 1 else "Your GPUs: ") + "; ".join(parts) + "." if parts else ""
    for n in nvidia:
        s += (f" NVIDIA {n['name']}: {n['vram_total_human']} VRAM, driver {n['driver_version']}"
              + (f", {n['temperature_c']}°C" if n.get("temperature_c") is not None else "") + ".")
    if r.get("unavailable"):
        s += f" Not available: {_unknown(r)}."
    return s.strip()


def say_battery(fetch: Fetch) -> str:
    d = fetch("get_battery_info").get("data", {})
    if not d.get("has_battery"):
        return "No battery detected - this looks like a desktop, server or VM."
    out = []
    for b in d["batteries"]:
        s = f"Battery {b['name']}: {b['charge_pct']}% charged ({b['status']})" if b.get("charge_pct") is not None \
            else f"Battery {b['name']} ({b['status']})"
        s += f", health {b['health_pct']}% of design capacity" if b.get("health_pct") else ", health not reported"
        if b.get("cycle_count"):
            s += f", {b['cycle_count']} charge cycles"
        out.append(s + ".")
    if d.get("ac_adapter_online") is not None:
        out.append(f"AC adapter is {'connected' if d['ac_adapter_online'] else 'not connected'}.")
    return " ".join(out)


def say_network(fetch: Fetch) -> str:
    d = fetch("get_network_interfaces").get("data", {})
    phys = [i for i in d.get("interfaces", []) if i["kind"] != "virtual"]
    if not phys:
        return "I couldn't find any physical network adapters."
    s = "Network adapters: " + "; ".join(
        f"{i['name']} ({i['kind']}, {i['state']}"
        + (f", {i['link_speed_mbps']} Mb/s" if i.get("link_speed_mbps") else "")
        + (f", driver {i['driver']}" if i.get("driver") else "") + ")" for i in phys) + "."
    if d.get("controllers"):
        s += " Controllers: " + "; ".join(d["controllers"]) + "."
    return s


def say_temps(fetch: Fetch) -> str:
    r = fetch("get_temperatures")
    sensors = r.get("data", {}).get("sensors", [])
    if not sensors:
        return f"I can't read temperatures ({_unknown(r)})."
    hottest = max(sensors, key=lambda x: x["celsius"])
    return ("Temperatures: " + ", ".join(f"{x['sensor']} {x['celsius']}°C" for x in sensors[:8])
            + f". Hottest: {hottest['sensor']} at {hottest['celsius']}°C.")


def say_overview(fetch: Fetch) -> str:
    r = fetch("get_system_overview")
    d = r.get("data", {})
    s = ""
    if d.get("manufacturer") or d.get("model"):
        s += f"This is a {' '.join(filter(None, [d.get('manufacturer'), d.get('model')]))}"
        s += f" ({d['form_factor']})." if d.get("form_factor") and d["form_factor"] != "unknown" else "."
        if d.get("bios"):
            s += f" BIOS: {d['bios']}."
    else:
        s += f"I can't determine the manufacturer or model ({_unknown(r)})."
    s += f" It runs {d.get('os', 'an unknown OS')} on {d.get('kernel')} ({d.get('architecture')})."
    if d.get("virtual_machine"):
        s += " It appears to be a virtual machine, so some hardware details reflect virtual devices."
    return s


COMPOSERS: dict[str, Callable[..., str]] = {
    "ram_upgrade": say_ram_upgrade, "ram": say_ram, "cpu": say_cpu, "storage": say_storage,
    "disk_usage": say_disk_usage, "gpu": say_gpu, "battery": say_battery, "network": say_network,
    "temps": say_temps, "overview": say_overview,
}


class OfflineEngine:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.last_intents: list[str] = []

    def answer(self, question: str, on_tool: Callable[[str, dict], None] | None = None) -> tuple[str, list[str]]:
        intents = route(question, self.last_intents)
        if not intents:
            return HELP, []
        self.last_intents = intents
        used: list[str] = []

        def fetch(tool: str) -> dict:
            result = self.registry.call(tool)
            if tool not in used:
                used.append(tool)
                if on_tool:
                    on_tool(tool, result)
            return result

        parts = []
        for name in intents:
            fn = COMPOSERS[name]
            parts.append(fn(fetch, question) if name == "storage" else fn(fetch))
        return "\n".join(parts), used
