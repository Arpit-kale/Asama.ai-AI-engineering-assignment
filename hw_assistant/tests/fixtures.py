"""Captured outputs from a typical laptop (16 GB in 1 of 2 slots, 64 GB max, NVMe, Intel + NVIDIA)."""
from __future__ import annotations

from hwqa.executor import CommandUnavailable, PermissionNeeded, SystemAccess

DMIDECODE_MEMORY = """# dmidecode 3.5
Getting SMBIOS data from sysfs.
SMBIOS 3.3.0 present.

Handle 0x0010, DMI type 16, 23 bytes
Physical Memory Array
\tLocation: System Board Or Motherboard
\tUse: System Memory
\tError Correction Type: None
\tMaximum Capacity: 64 GB
\tError Information Handle: Not Provided
\tNumber Of Devices: 2

Handle 0x0011, DMI type 17, 92 bytes
Memory Device
\tArray Handle: 0x0010
\tTotal Width: 64 bits
\tData Width: 64 bits
\tSize: 16 GB
\tForm Factor: SODIMM
\tSet: None
\tLocator: DIMM A
\tBank Locator: BANK 0
\tType: DDR4
\tType Detail: Synchronous
\tSpeed: 3200 MT/s
\tManufacturer: Samsung
\tSerial Number: 1234ABCD
\tPart Number: M471A2K43DB1-CWE
\tRank: 2
\tConfigured Memory Speed: 3200 MT/s

Handle 0x0012, DMI type 17, 92 bytes
Memory Device
\tArray Handle: 0x0010
\tTotal Width: Unknown
\tData Width: Unknown
\tSize: No Module Installed
\tForm Factor: Unknown
\tSet: None
\tLocator: DIMM B
\tBank Locator: BANK 2
\tType: Unknown
\tType Detail: None
\tSpeed: Unknown
\tManufacturer: Not Specified
"""

LSCPU_JSON = """{
   "lscpu": [
      {"field": "Architecture:", "data": "x86_64",
       "children": [{"field": "CPU op-mode(s):", "data": "32-bit, 64-bit"}]},
      {"field": "CPU(s):", "data": "8"},
      {"field": "Vendor ID:", "data": "GenuineIntel",
       "children": [
          {"field": "Model name:", "data": "11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz",
           "children": [
              {"field": "Thread(s) per core:", "data": "2"},
              {"field": "Core(s) per socket:", "data": "4"},
              {"field": "Socket(s):", "data": "1"},
              {"field": "CPU max MHz:", "data": "4700.0000"},
              {"field": "CPU min MHz:", "data": "400.0000"},
              {"field": "Flags:", "data": "fpu vme sse4_2 avx avx2 avx512f vmx"}
           ]}
       ]},
      {"field": "Virtualization features:", "data": null,
       "children": [{"field": "Virtualization:", "data": "VT-x"}]},
      {"field": "Caches (sum of all):", "data": null,
       "children": [{"field": "L2:", "data": "5 MiB (4 instances)"},
                    {"field": "L3:", "data": "12 MiB (1 instance)"}]}
   ]
}"""

FREE_B = """               total        used        free      shared  buff/cache   available
Mem:     16541196288  5368709120  6442450944   536870912  4730036224 10737418240
Swap:     2147483648           0  2147483648
"""

LSBLK_JSON = """{
   "blockdevices": [
      {"name":"nvme0n1", "size":512110190592, "type":"disk", "model":"Samsung SSD 980 PRO 512GB",
       "vendor":null, "rota":false, "tran":"nvme", "mountpoint":null, "fstype":null, "rm":false,
       "children": [
          {"name":"nvme0n1p1", "size":536870912, "type":"part", "mountpoint":"/boot/efi", "rota":false},
          {"name":"nvme0n1p2", "size":511571206144, "type":"part", "mountpoint":"/", "rota":false}
       ]},
      {"name":"sda", "size":"1000204886016", "type":"disk", "model":"WDC WD10SPZX-21Z", "vendor":"ATA     ",
       "rota":"1", "tran":"sata", "mountpoint":"/data", "fstype":"ext4", "rm":"0"},
      {"name":"zram0", "size":8589934592, "type":"disk", "rota":false, "mountpoint":"[SWAP]"}
   ]
}"""

LSPCI_MM = '''00:00.0 "Host bridge [0600]" "Intel Corporation [8086]" "11th Gen Core Processor Host Bridge [9a14]" -r01 "Lenovo [17aa]" "Device [22d8]"
00:02.0 "VGA compatible controller [0300]" "Intel Corporation [8086]" "TigerLake-LP GT2 [Iris Xe Graphics] [9a49]" -r01 "Lenovo [17aa]" "Device [22d8]"
01:00.0 "3D controller [0302]" "NVIDIA Corporation [10de]" "GA107M [GeForce RTX 3050 Mobile] [25a2]" -ra1 "Lenovo [17aa]" "Device [22d8]"
00:14.3 "Network controller [0280]" "Intel Corporation [8086]" "Wi-Fi 6 AX201 [a0f0]" -r20 "Intel Corporation [8086]" "Wi-Fi 6 AX201 160MHz [0070]"
'''

NVIDIA_SMI = "NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 5, 535.129.03, 45, 0\n"

OUTPUTS = {"dmidecode_memory": DMIDECODE_MEMORY, "lscpu_json": LSCPU_JSON, "free": FREE_B,
           "lsblk": LSBLK_JSON, "lspci": LSPCI_MM, "nvidia_smi": NVIDIA_SMI}

FILES = {
    "/etc/os-release": 'PRETTY_NAME="Ubuntu 24.04 LTS"\nNAME="Ubuntu"\n',
    "/sys/class/dmi/id/sys_vendor": "LENOVO",
    "/sys/class/dmi/id/product_name": "20XW0055US",
    "/sys/class/dmi/id/product_version": "ThinkPad X1 Carbon Gen 9",
    "/sys/class/dmi/id/chassis_type": "10",
    "/sys/class/dmi/id/bios_version": "N32ET75W (1.51 )",
    "/proc/meminfo": "MemTotal:       16153512 kB\nMemFree:         6291456 kB\nMemAvailable:   10485760 kB\n",
    "/proc/cpuinfo": "processor\t: 0\nflags\t\t: fpu avx2\n",
    "/sys/class/power_supply/BAT0/type": "Battery",
    "/sys/class/power_supply/BAT0/status": "Discharging",
    "/sys/class/power_supply/BAT0/capacity": "76",
    "/sys/class/power_supply/BAT0/energy_full": "47520000",
    "/sys/class/power_supply/BAT0/energy_full_design": "57000000",
    "/sys/class/power_supply/BAT0/cycle_count": "212",
    "/sys/class/power_supply/AC/type": "Mains",
    "/sys/class/power_supply/AC/online": "0",
}
DIRS = {"/sys/class/power_supply": ["AC", "BAT0"]}


class FakeSystem(SystemAccess):
    """Canned system: lets tests run identically on any machine (or CI container)."""

    def __init__(self, outputs=None, files=None, root: bool = False, missing=(), needs_root=("dmidecode_memory",)):
        self.outputs = dict(OUTPUTS if outputs is None else outputs)
        self.files = dict(FILES if files is None else files)
        self.root, self.missing, self.needs_root = root, set(missing), set(needs_root)
        self.calls: list[str] = []

    def available(self, command_id):
        return command_id in self.outputs and command_id not in self.missing

    def run(self, command_id):
        self.calls.append(command_id)
        if not self.available(command_id):
            raise CommandUnavailable(command_id)
        if command_id in self.needs_root and not self.root:
            raise PermissionNeeded(f"{command_id} needs root privileges")
        return self.outputs[command_id]

    def read_text(self, path):
        return self.files.get(path)

    def list_dir(self, path):
        return DIRS.get(path, [])

    def link_name(self, path):
        return None

    def is_root(self):
        return self.root
