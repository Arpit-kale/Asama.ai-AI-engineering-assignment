#!/usr/bin/env bash
# hwqa installer
#
#   ./install.sh                 install the `hwqa` command for the current user
#   ./install.sh --sudo-rule     ... and add the scoped sudoers rule for dmidecode (asks first)
#   ./install.sh --ollama        ... and pull a local model if Ollama is installed
#   ./install.sh --uninstall     remove hwqa and the sudoers rule
#
# Installs from this folder, or from a wheel:  ./install.sh --from dist/hwqa-1.1.0-py3-none-any.whl
set -euo pipefail

SUDOERS_FILE=/etc/sudoers.d/hwqa
MODEL="${HWQA_OLLAMA_MODEL:-qwen2.5:7b-instruct}"
SRC="$(cd "$(dirname "$0")" && pwd)"
WANT_SUDO=0 WANT_OLLAMA=0 UNINSTALL=0

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31mxx\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sudo-rule) WANT_SUDO=1 ;;
    --ollama)    WANT_OLLAMA=1 ;;
    --uninstall) UNINSTALL=1 ;;
    --from)      SRC="$(realpath "$2")"; shift ;;
    -h|--help)   sed -n '2,10p' "$0"; exit 0 ;;
    *)           die "unknown option: $1" ;;
  esac
  shift
done

[[ "$(uname -s)" == "Linux" ]] || die "hwqa inspects Linux hardware; this is $(uname -s)."
command -v python3 >/dev/null || die "python3 not found (install python3 >= 3.9)."
python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' || die "python3 >= 3.9 required."
[[ $EUID -ne 0 ]] || warn "Running as root installs hwqa for root. Prefer your normal user; the sudoers rule covers dmidecode."

if [[ $UNINSTALL -eq 1 ]]; then
  say "Removing hwqa"
  if command -v pipx >/dev/null && pipx list 2>/dev/null | grep -q 'package hwqa'; then
    pipx uninstall hwqa
  else
    python3 -m pip uninstall -y hwqa 2>/dev/null || true
  fi
  if [[ -f $SUDOERS_FILE ]]; then
    say "Removing $SUDOERS_FILE"
    sudo rm -f "$SUDOERS_FILE"
  fi
  say "Done."
  exit 0
fi

# ---------------------------------------------------------------- install the command
say "Installing hwqa from $SRC"
if command -v pipx >/dev/null; then
  pipx install --force "$SRC"
else
  warn "pipx not found; using 'pip install --user' (install pipx for isolated installs)."
  python3 -m pip install --user --upgrade "$SRC" 2>/dev/null \
    || python3 -m pip install --user --upgrade --break-system-packages "$SRC"
fi

BIN="$(command -v hwqa || true)"
[[ -n $BIN ]] || BIN="$HOME/.local/bin/hwqa"
[[ -x $BIN ]] || die "install finished but the hwqa command was not found"
case ":$PATH:" in *":$(dirname "$BIN"):"*) ;; *) warn "Add $(dirname "$BIN") to your PATH (e.g. 'pipx ensurepath')." ;; esac
say "Installed $("$BIN" --version) at $BIN"

# ---------------------------------------------------------------- optional: sudoers rule
if [[ $WANT_SUDO -eq 1 ]]; then
  command -v dmidecode >/dev/null || command -v /usr/sbin/dmidecode >/dev/null \
    || warn "dmidecode is not installed yet (apt/dnf/pacman install dmidecode); the rule is written anyway."
  RULE="$("$BIN" --print-sudoers-rule)"
  echo
  echo "This allows ONE exact read-only command without a password:"
  echo "    $RULE"
  read -r -p "Write it to $SUDOERS_FILE? [y/N] " ok
  if [[ ${ok,,} == y* ]]; then
    tmp="$(mktemp)"
    printf '%s\n' "$RULE" > "$tmp"
    sudo visudo -cf "$tmp" >/dev/null || { rm -f "$tmp"; die "sudoers syntax check failed; nothing written"; }
    sudo install -m 0440 -o root -g root "$tmp" "$SUDOERS_FILE"
    rm -f "$tmp"
    say "Sudoers rule installed. Test: sudo -n $(echo "$RULE" | sed 's/.*NOPASSWD: //')"
  else
    say "Skipped. hwqa will report root-only details as unavailable (or use: hwqa --sudo prompt)."
  fi
fi

# ---------------------------------------------------------------- optional: local LLM
if [[ $WANT_OLLAMA -eq 1 ]]; then
  if command -v ollama >/dev/null; then
    say "Pulling local model $MODEL (several GB, one time)"
    ollama pull "$MODEL"
  else
    warn "Ollama is not installed. Get it from https://ollama.com/download, then: ollama pull $MODEL"
    warn "hwqa works without it (offline engine)."
  fi
fi

echo
say "Ready. Try:  hwqa        or   hwqa -q \"Can I add more RAM?\"      (hwqa --check shows available tools)"
