"""Command-line chat interface.

    python -m hwqa                      # interactive chat (auto-detects a local Ollama model)
    python -m hwqa -q "Is my SSD NVMe?" # one-shot
    python -m hwqa --check              # show which system tools are available
    python -m hwqa --setup-sudo         # print a least-privilege sudoers rule for dmidecode
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .agent import Assistant
from .executor import COMMANDS, SafeExecutor, SudoMode, sudoers_rule, which_report
from .llm import build_client
from .tools import ToolRegistry

try:
    import readline  # noqa: F401  (arrow keys / history in input())
except ImportError:
    pass

_TTY = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


BANNER = """Linux Hardware Q&A - ask about CPU, RAM, disks, GPU, battery, network, temperatures.
Commands: /tools  /facts  /debug  /clear  /help  /quit"""


def setup_sudo_text() -> str:
    rule = sudoers_rule()
    return f"""Some details (RAM slots, max supported RAM, module type/speed) come from `dmidecode`,
which needs root. Instead of running this assistant as root, allow ONLY that exact command:

  echo '{rule}' | sudo tee /etc/sudoers.d/hwqa
  sudo chmod 440 /etc/sudoers.d/hwqa && sudo visudo -cf /etc/sudoers.d/hwqa

The rule pins the binary path AND its arguments, so nothing else can be run through it.
Alternatives: start with `--sudo prompt` (asks for your password once), or `--sudo never`
(the assistant then reports those details as unavailable instead of guessing)."""


def check_text(executor: SafeExecutor) -> str:
    lines = ["System capability check (binaries resolved from trusted dirs only):"]
    for binary, ok in which_report().items():
        lines.append(f"  {'✓' if ok else '✗'} {binary}")
    lines.append(f"  running as root: {executor.is_root()}")
    lines.append(f"  root-only commands: {', '.join(k for k, v in COMMANDS.items() if v.needs_root)} "
                 f"(sudo mode: {executor.sudo_mode.value})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hwqa", description="Local natural-language hardware assistant for Linux.")
    ap.add_argument("-q", "--question", help="ask one question and exit")
    ap.add_argument("--llm", default="auto", choices=["auto", "ollama", "openai-compatible", "openai", "offline"],
                    help="auto = local Ollama if running, else offline engine (default)")
    ap.add_argument("--model", help="model name, e.g. qwen2.5:7b-instruct")
    ap.add_argument("--sudo", default="noninteractive", choices=[m.value for m in SudoMode],
                    help="how to run root-only commands (default: sudo -n, needs --setup-sudo rule)")
    ap.add_argument("--debug", action="store_true", help="show tool calls and their JSON")
    ap.add_argument("--check", action="store_true", help="print available system tools and exit")
    ap.add_argument("--setup-sudo", action="store_true", help="explain and print the scoped sudoers rule, then exit")
    ap.add_argument("--print-sudoers-rule", action="store_true", help=argparse.SUPPRESS)  # used by install.sh
    ap.add_argument("--version", action="version", version=f"hwqa {__version__}")
    args = ap.parse_args(argv)

    if args.print_sudoers_rule:
        print(sudoers_rule())
        return 0
    if args.setup_sudo:
        print(setup_sudo_text())
        return 0
    executor = SafeExecutor(SudoMode(args.sudo))
    if args.check:
        print(check_text(executor))
        return 0

    debug = {"on": args.debug}

    def on_tool(name: str, call_args: dict, result: dict) -> None:
        if debug["on"]:
            arg_txt = json.dumps(call_args) if call_args else ""
            print(c(f"  ⚙ {name}({arg_txt})", "2"))
            print(c("    " + json.dumps(result, ensure_ascii=False)[:1500], "2"))

    registry = ToolRegistry(executor)
    client, desc = build_client(args.llm, args.model) if args.llm != "offline" else (None, "offline deterministic engine")
    bot = Assistant(registry, client, on_tool)

    if args.question:
        print(bot.ask(args.question).text)
        return 0

    print(c(BANNER, "1"))
    privacy = "all data stays on this machine" if (client is None or client.is_local) else "data is sent to the cloud LLM"
    print(c(f"Engine: {desc} - {privacy}.\n", "2"))
    while True:
        try:
            q = input(c("you › ", "36")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        cmd = q.lower()
        if cmd in ("/quit", "/exit", "quit", "exit"):
            return 0
        if cmd == "/help":
            print(BANNER)
            continue
        if cmd == "/tools":
            for name, tool in registry.tools.items():
                print(f"  {c(name, '1')}: {tool.description}")
            continue
        if cmd == "/facts":
            print(json.dumps(bot.session_results, indent=2, ensure_ascii=False)[:8000] or "  (nothing gathered yet)")
            continue
        if cmd == "/debug":
            debug["on"] = not debug["on"]
            print(f"  debug {'on' if debug['on'] else 'off'}")
            continue
        if cmd == "/clear":
            bot.reset()
            print("  conversation and cache cleared")
            continue
        ans = bot.ask(q)
        print(c("bot › ", "32") + ans.text + "\n")


if __name__ == "__main__":
    sys.exit(main())
