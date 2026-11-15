# Problem 2: Hardware Q&A assistant for your Linux laptop

A local chatbot that answers plain-language questions about this machine's hardware. It never guesses.
An LLM, or the built-in offline engine, calls a set of **allow-listed, read-only inspection tools**. Their
output is parsed into structured JSON, and the answer is built only from that data.

```
you › Can I add more RAM, and how much?
bot › You have 16 GB in 1 of 2 slots. The board supports up to 64 GB, so you can add a stick of up to 32 GB.
      Use the same type as your current module (DDR4 SODIMM).
you › and what about the GPU?
bot › Your GPUs: Intel Corporation TigerLake-LP GT2 [Iris Xe Graphics] (integrated); NVIDIA Corporation
      GA107M [GeForce RTX 3050 Mobile] (discrete). NVIDIA GeForce RTX 3050 Laptop GPU: 4 GB VRAM, driver 535.129.03, 45°C.
```

## Install

```bash
./install.sh                    # installs the `hwqa` command (pipx, or pip --user as a fallback)
./install.sh --sudo-rule        # ... and the scoped dmidecode sudoers rule (asks first)
./install.sh --uninstall
./scripts/build.sh              # build dist/hwqa-*.whl and dist/hwqa.pyz (one file, runs anywhere)
```

See [../DEPLOYMENT.md](../DEPLOYMENT.md) for release downloads.

## Run it

```bash
python3 -m hwqa                        # interactive chat (no dependencies: standard library only)
python3 -m hwqa -q "Is my SSD NVMe?"   # one-shot question
python3 -m hwqa --check                # which system tools exist on this machine
python3 -m hwqa --setup-sudo           # print the least-privilege sudoers rule for dmidecode
python3 -m hwqa --debug                # show every tool call and its JSON
pip install -e . && hwqa               # optional: install the `hwqa` command

pip install pytest && python3 -m pytest -q   # 35 tests, using captured real-laptop output
```

In chat you can use `/tools`, `/facts` (the data gathered so far), `/debug`, `/clear` and `/quit`.

### LLM backends (`--llm`)

| Value | What it does | Privacy |
|---|---|---|
| `auto` (default) | Uses Ollama if it's running locally, otherwise the offline engine. | Nothing leaves the machine. |
| `ollama` | Uses a local model, e.g. `ollama pull qwen2.5:7b-instruct` then `--model qwen2.5:7b-instruct`. | Local. |
| `openai-compatible` | Uses a llama.cpp server or vLLM (`HWQA_LLM_BASE_URL`, `HWQA_LLM_MODEL`). | Local. |
| `openai` | Uses the OpenAI API (`OPENAI_API_KEY`). The banner warns that data leaves the machine. | Cloud, opt-in. |
| `offline` | Uses the deterministic rules-and-templates engine. No model is needed. | Local. |

If the LLM is unreachable or keeps looping on tool calls, the assistant answers with the offline engine instead of failing.

## Architecture

```
CLI (cli.py) ─▶ Assistant (agent.py) ──▶ LLM tool-calling loop (llm.py, OpenAI-compatible)
                       │                 └─ or OfflineEngine (offline.py): intent router + answer templates
                       ▼
               ToolRegistry (tools.py)   10 tools · JSON-schema arg validation · TTL cache · fallbacks
                       ▼
               SafeExecutor (executor.py) allow-listed argv · no shell · trusted paths · sudo -n · file allow-list
                       ▼
               parsers.py                pure functions: lscpu/lsblk/free/lspci/dmidecode/nvidia-smi → JSON
```

| Tool | Primary source | Fallback |
|---|---|---|
| `get_system_overview` | `/sys/class/dmi/id`, `/etc/os-release`, `uname` | Reports each field as unavailable. |
| `get_cpu_info` | `lscpu -J` | `lscpu` text, then `/proc/cpuinfo`. |
| `get_memory_info` | `free -b` | `/proc/meminfo`. |
| `get_memory_slots` | `dmidecode -t memory` (root) plus a pre-computed upgrade analysis | Usable RAM from `/proc/meminfo`, with the reason the rest is unavailable. |
| `get_storage_devices` | `lsblk -J -b` | `/sys/block`. |
| `get_disk_usage` | `/proc/mounts` + `statvfs` | |
| `get_gpu_info` | `lspci -mm -nn`, `nvidia-smi` | `/sys/class/drm` (PCI IDs only). |
| `get_network_interfaces` | `/sys/class/net`, `lspci` | |
| `get_battery_info` | `/sys/class/power_supply` | |
| `get_temperatures` | `/sys/class/thermal`, `/sys/class/hwmon` | |

## Design choices

**Safety (allow-list, no shell).**
- The LLM only picks a tool name. Every command is a fixed `argv` in `executor.COMMANDS`, so no text from the LLM ever reaches `execve`.
- Binaries are resolved from trusted system directories, not `$PATH`, into absolute paths.
- Commands run with `shell=False`, a minimal `LC_ALL=C` environment, closed stdin, timeouts and an output cap.
- Tool arguments are validated against a JSON schema. For example, `device` must match `[a-zA-Z0-9]{1,32}` and is only used to filter results.
- File reads are limited to `/sys` and a few named `/proc` and `/etc` files, checked after resolving symlinks. `/etc/shadow`, `/proc/self/environ` and `..` tricks are rejected, and tests cover these cases.

**Permissions.** The assistant never runs as root. Root-only data, which is only `dmidecode`, is handled three ways:
1. `sudo -n` (the default) works only with the scoped rule that `--setup-sudo` prints:
   `user ALL=(root) NOPASSWD: /usr/sbin/dmidecode -t memory`. The rule pins the binary **and its arguments**. I checked this end to end: the exact command runs, while `dmidecode -t system` and `ls` are refused.
2. `--sudo prompt` asks for the password once in the terminal.
3. `--sudo never`, or a missing rule, means the assistant gives a partial, honest answer: "I can't determine whether you can add RAM… needs root… What I do know: 15.4 GB is usable."

**Accuracy (no hallucination).**
- The system prompt forbids stating facts that aren't in a tool result.
- Each tool returns `data`, `sources` and `unavailable` (with reasons), so "I don't know" is backed by data.
- Derived numbers, such as the largest RAM stick you can add, are computed in Python and not left to the model.
- A grounding check flags any number with a unit in the LLM's answer that can't be traced to tool output at any unit scale (for example "You can upgrade to 128 GB" when the data says 64 GB).
- The offline engine builds every sentence from tool data.

**Portability.**
- Every tool has fallbacks for missing binaries: `/proc` and `/sys` are always tried.
- The parsers handle differences between util-linux versions: nested `lscpu -J` output, `rota` given as a bool or `"0"`/`"1"`, and sizes given as numbers or strings.
- Virtual machines are detected and explained ("virtual disk", "running under KVM", "no SMBIOS tables").
- `--check` shows what is available on the machine.

**Follow-ups.**
- The LLM gets the trimmed conversation, including earlier tool results.
- The offline router remembers the last topic. "How fast is it?" after a CPU question resolves to the CPU, while unrelated questions get a help message instead of being treated as follow-ups.
- Tool results are cached with a TTL: static facts for an hour, live metrics (memory use, temperatures) for a few seconds.

**Privacy.**
- By default everything runs locally: the offline engine or Ollama.
- Serial numbers are never collected.
- Cloud LLM use is opt-in and announced in the banner.
