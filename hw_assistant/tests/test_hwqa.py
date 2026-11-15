from __future__ import annotations

import json

import pytest

from hwqa import parsers as P
from hwqa.agent import Assistant, ungrounded_numbers
from hwqa.executor import COMMANDS, SafeExecutor, SudoMode, PermissionNeeded, sudoers_rule
from hwqa.llm import LLMReply, ToolCall
from hwqa.offline import HELP, route
from hwqa.tools import ToolRegistry, validate_args, ToolArgumentError

from .fixtures import DMIDECODE_MEMORY, LSBLK_JSON, LSCPU_JSON, LSPCI_MM, FakeSystem

GB = 1024**3


# =========================================================================== parsers
def test_dmidecode_and_upgrade_math():
    topo = P.parse_dmidecode_memory(DMIDECODE_MEMORY)
    assert topo["max_capacity_bytes"] == 64 * GB
    assert (topo["total_slots"], topo["used_slots"], topo["free_slots"]) == (2, 1, 1)
    assert topo["installed_bytes"] == 16 * GB
    assert topo["slots"][0]["type"] == "DDR4" and "serial" not in json.dumps(topo).lower()
    up = P.upgrade_analysis(topo)
    assert up["max_per_slot_bytes"] == 32 * GB
    assert up["max_addable_now_bytes"] == 32 * GB
    assert up["headroom_bytes"] == 48 * GB and up["upgrade_possible"] is True


def test_upgrade_all_slots_full_and_at_max():
    full = {"max_capacity_bytes": 32 * GB, "installed_bytes": 16 * GB, "total_slots": 2, "free_slots": 0,
            "slots": [{"installed": True, "size_bytes": 8 * GB, "soldered": False}] * 2}
    up = P.upgrade_analysis(full)
    assert up["max_addable_now_bytes"] == 0 and up["upgrade_possible"] is True     # replace modules
    maxed = {**full, "installed_bytes": 32 * GB}
    assert P.upgrade_analysis(maxed)["upgrade_possible"] is False


def test_lscpu_nested_json():
    cpu = P.normalise_cpu(P.parse_lscpu_json(LSCPU_JSON))
    assert cpu["model"].startswith("11th Gen Intel")
    assert (cpu["physical_cores"], cpu["logical_cpus"], cpu["threads_per_core"]) == (4, 8, 2)
    assert cpu["max_mhz"] == 4700.0 and cpu["l3_cache"] is None or cpu["l3_cache"].startswith("12 MiB")
    assert cpu["supports_avx512"] is True


def test_lsblk_classifies_disks():
    disks = {d["name"]: d for d in P.parse_lsblk_json(LSBLK_JSON)}
    assert set(disks) == {"nvme0n1", "sda"}                                  # zram skipped
    assert disks["nvme0n1"]["is_nvme"] and disks["nvme0n1"]["mountpoints"] == ["/boot/efi", "/"]
    assert disks["sda"]["kind"] == "HDD (rotational)" and disks["sda"]["vendor"] == "ATA"


def test_lspci_gpus():
    gpus = P.gpus_from_lspci(P.parse_lspci_mm(LSPCI_MM))
    assert [g["vendor_id"] for g in gpus] == ["8086", "10de"]
    assert gpus[1]["device"] == "GA107M [GeForce RTX 3050 Mobile]"


@pytest.mark.parametrize("text,expected", [("16 GB", 16 * GB), ("8192 MB", 8 * GB), ("512 kB", 512 * 1024),
                                           ("No Module Installed", None), ("", None)])
def test_parse_size(text, expected):
    assert P.parse_size(text) == expected


# =========================================================================== security
def test_only_allow_listed_commands_can_run():
    ex = SafeExecutor(SudoMode.NEVER)
    with pytest.raises(PermissionError):
        ex.build_argv("rm -rf /")
    with pytest.raises(PermissionError):
        ex.build_argv("lscpu; rm -rf /")
    for spec in COMMANDS.values():                          # read-only binaries only
        assert spec.binary in {"lscpu", "free", "lsblk", "lspci", "dmidecode", "nvidia-smi"}


def test_argv_is_absolute_and_shell_free():
    ex = SafeExecutor(SudoMode.NEVER)
    if not ex.available("free"):
        pytest.skip("free not installed")
    argv = ex.build_argv("free")
    assert argv[0].startswith("/") and argv[1:] == ["-b"]
    assert "free" in ex.run("free").lower() or "mem:" in ex.run("free").lower()


def test_root_command_without_privilege(monkeypatch):
    ex = SafeExecutor(SudoMode.NEVER)
    monkeypatch.setattr(ex, "is_root", lambda: False)
    ex._paths["dmidecode_memory"] = "/usr/sbin/dmidecode"
    with pytest.raises(PermissionNeeded):
        ex.build_argv("dmidecode_memory")
    ex.sudo_mode, ex._sudo = SudoMode.NONINTERACTIVE, "/usr/bin/sudo"
    assert ex.build_argv("dmidecode_memory") == ["/usr/bin/sudo", "-n", "/usr/sbin/dmidecode", "-t", "memory"]


def test_sudoers_rule_pins_arguments():
    assert "NOPASSWD:" in sudoers_rule() and sudoers_rule().endswith("dmidecode -t memory")


@pytest.mark.parametrize("path", ["/etc/shadow", "/root/.ssh/id_rsa", "/sys/../etc/passwd",
                                  "/proc/self/environ", "/proc/1/cmdline", "/home/x/.bashrc"])
def test_file_reads_are_restricted(path):
    with pytest.raises(PermissionError):
        SafeExecutor().read_text(path)


def test_tool_argument_validation():
    reg = ToolRegistry(FakeSystem())
    assert "invalid arguments" in reg.call("get_storage_devices", {"device": "sda; rm -rf /"})["error"]
    assert "unknown argument" in reg.call("get_cpu_info", {"cmd": "whoami"})["error"]
    assert "unknown tool" in reg.call("run_shell", {"cmd": "ls"})["error"]
    with pytest.raises(ToolArgumentError):
        validate_args({"properties": {"x": {"type": "boolean"}}}, {"x": "maybe"})
    assert reg.call("get_storage_devices", {"device": "sda"})["data"]["count"] == 1


# =========================================================================== tools
def test_memory_slots_root_vs_unprivileged():
    root = ToolRegistry(FakeSystem(root=True)).call("get_memory_slots")
    assert root["data"]["upgrade"]["max_addable_now_human"] == "32 GB"
    user = ToolRegistry(FakeSystem(root=False)).call("get_memory_slots")
    assert user["unavailable"][0]["reason"].startswith("needs root")
    assert user["data"]["usable_total_human"] == "15.4 GB"            # still a partial, honest answer


def test_missing_tools_fall_back():
    sys_ = FakeSystem(missing={"lscpu_json", "lspci", "lsblk", "free", "nvidia_smi"},
                      files={**FakeSystem().files,
                             "/proc/cpuinfo": "processor\t: 0\nmodel name\t: Test CPU\nphysical id\t: 0\ncpu cores\t: 2\n\n"
                                              "processor\t: 1\nmodel name\t: Test CPU\nphysical id\t: 0\ncpu cores\t: 2\n"})
    reg = ToolRegistry(sys_)
    cpu = reg.call("get_cpu_info")
    assert cpu["data"]["model"] == "Test CPU" and cpu["sources"] == ["/proc/cpuinfo"]
    assert reg.call("get_memory_info")["data"]["total_human"] == "15.4 GB"
    assert reg.call("get_gpu_info")["unavailable"]


def test_tool_cache():
    sys_ = FakeSystem(root=True)
    reg = ToolRegistry(sys_)
    reg.call("get_cpu_info")
    reg.call("get_cpu_info")
    assert sys_.calls.count("lscpu_json") == 1


# =========================================================================== offline engine
@pytest.fixture
def bot():
    return Assistant(ToolRegistry(FakeSystem(root=True)))


def test_example_ram_upgrade_answer(bot):
    ans = bot.ask("Can I add more RAM, and how much?")
    assert ans.text.startswith("You have 16 GB in 1 of 2 slots. The board supports up to 64 GB, "
                               "so you can add a stick of up to 32 GB.")
    assert "get_memory_slots" in ans.tools_used


def test_ram_nvme_and_followups(bot):
    assert "16 GB of RAM installed (DDR4, 3200 MT/s), using 1 of 2 slots" in bot.ask("How much RAM do I have?").text
    assert bot.ask("Is my SSD NVMe?").text.startswith("Yes, nvme0n1 is an NVMe SSD.")
    gpu = bot.ask("and what about the GPU?").text
    assert "Iris Xe" in gpu and "RTX 3050" in gpu and "4 GB VRAM" in gpu
    bot.ask("how many cores does my cpu have?")
    assert "4.70 GHz" in bot.ask("how fast is it?").text       # follow-up resolved to CPU


def test_battery_overview_and_unknowns(bot):
    assert "76% charged" in bot.ask("battery health?").text and "83.4%" in bot.ask("battery health?").text
    assert "ThinkPad X1 Carbon Gen 9" in bot.ask("what laptop model is this?").text
    assert bot.ask("what's the weather tomorrow").text == HELP


def test_unprivileged_says_i_dont_know():
    bot = Assistant(ToolRegistry(FakeSystem(root=False)))
    text = bot.ask("Can I upgrade my RAM?").text
    assert text.startswith("I can't determine whether you can add RAM")
    assert "64" not in text                                     # never invents the max capacity


def test_router():
    assert route("How much memory is installed", []) == ["ram"]
    assert route("how many ram slots", []) == ["ram_upgrade"]
    assert route("how much disk space is left", []) == ["disk_usage"]
    assert route("and that one?", ["gpu"]) == ["gpu"]


# =========================================================================== LLM loop
class ScriptedLLM:
    """Fake model: first asks for a tool, then answers from the tool result."""
    model = "fake"
    is_local = True

    def __init__(self, answer: str, tool: str = "get_memory_slots"):
        self.answer, self.tool, self.seen = answer, tool, []

    def chat(self, messages, tools):
        self.seen.append(messages)
        if messages[-1]["role"] == "user":
            return LLMReply("", [ToolCall("c1", self.tool, {})])
        return LLMReply(self.answer)


def test_llm_tool_loop_and_grounding():
    llm = ScriptedLLM("You have 16 GB in 1 of 2 slots; the board supports up to 64 GB, so add up to 32 GB.")
    bot = Assistant(ToolRegistry(FakeSystem(root=True)), llm)
    ans = bot.ask("Can I add more RAM?")
    assert ans.engine == "llm:fake" and ans.tools_used == ["get_memory_slots"] and not ans.ungrounded
    tool_msg = llm.seen[1][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["data"]["free_slots"] == 1
    assert llm.seen[0][0]["role"] == "system" and "NEVER guess" in llm.seen[0][0]["content"]


def test_hallucinated_numbers_are_flagged():
    llm = ScriptedLLM("You can upgrade to 128 GB of RAM.")
    ans = Assistant(ToolRegistry(FakeSystem(root=True)), llm).ask("max RAM?")
    assert ans.ungrounded == ["128 GB"] and "couldn't verify" in ans.text


def test_follow_up_history_is_sent_to_llm():
    llm = ScriptedLLM("The GPU is an Intel Iris Xe.", tool="get_gpu_info")
    bot = Assistant(ToolRegistry(FakeSystem(root=True)), llm)
    bot.ask("How much RAM?")
    bot.ask("and what about the GPU?")
    roles = [m["role"] for m in llm.seen[-2]]
    assert roles.count("user") == 2 and "tool" in roles


def test_llm_failure_falls_back_to_offline():
    from hwqa.llm import LLMError

    class Down:
        model, is_local = "x", True

        def chat(self, *a):
            raise LLMError("connection refused")

    ans = Assistant(ToolRegistry(FakeSystem(root=True)), Down()).ask("How much RAM do I have?")
    assert ans.engine == "offline" and "16 GB" in ans.text and "LLM unavailable" in ans.text


def test_ungrounded_unit_scaling():
    facts = [{"data": {"total_bytes": 16541196288, "max_mhz": 4700.0}}]
    assert ungrounded_numbers("You have 15.4 GB and the CPU boosts to 4.7 GHz.", facts) == []
    assert ungrounded_numbers("You have 64 GB.", facts) == ["64 GB"]


def test_chat_client_over_http():
    """Real HTTP round-trip against an OpenAI-compatible server (as Ollama/vLLM/llama.cpp expose)."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from hwqa.llm import ChatClient

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            if body["messages"][-1]["role"] == "user":
                msg = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_storage_devices", "arguments": '{"device": "nvme0n1"}'}}]}
            else:
                msg = {"role": "assistant", "content": "Yes, nvme0n1 is an NVMe SSD (512.1 GB)."}
            out = json.dumps({"choices": [{"message": msg}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "qwen2.5:7b")
        assert client.is_local
        ans = Assistant(ToolRegistry(FakeSystem(root=True)), client).ask("Is my SSD NVMe?")
        assert ans.text == "Yes, nvme0n1 is an NVMe SSD (512.1 GB)." and not ans.ungrounded
        assert requests[0]["tools"][0]["type"] == "function" and requests[0]["model"] == "qwen2.5:7b"
        tool_msg = requests[1]["messages"][-1]
        assert tool_msg["tool_call_id"] == "call_1"
        assert json.loads(tool_msg["content"])["data"]["disks"][0]["name"] == "nvme0n1"
    finally:
        server.shutdown()
