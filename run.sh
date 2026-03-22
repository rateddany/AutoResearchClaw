#!/usr/bin/env bash
# ResearchClaw launcher — choose between Claude or Codex backend (via ACP/subscription)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

usage() {
    cat <<EOF
Usage: ./run.sh <backend> [options] [researchclaw args...]

Backends:
  claude    Use Claude Code (via your subscription)
  codex     Use Codex (via your subscription)

Options:
  --no-slurm    Run experiments locally (sandbox mode) instead of Slurm
  --slurm       Run experiments on Slurm cluster (default)

Examples:
  ./run.sh claude --topic "Neural scaling laws" --auto-approve
  ./run.sh codex  --topic "Diffusion model optimization" --auto-approve
  ./run.sh claude --no-slurm --topic "Quick local test" --auto-approve
  ./run.sh claude --resume
  ./run.sh codex  doctor

EOF
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

BACKEND="$1"
shift

case "$BACKEND" in
    claude)
        if ! command -v claude &>/dev/null; then
            echo "Error: 'claude' CLI not found on PATH."
            exit 1
        fi
        CONFIG="config.claude.yaml"
        echo "[ResearchClaw] Backend: Claude Code (ACP)"
        ;;
    codex)
        if ! command -v codex &>/dev/null; then
            echo "Error: 'codex' CLI not found on PATH."
            exit 1
        fi
        CONFIG="config.codex.yaml"
        echo "[ResearchClaw] Backend: Codex (ACP)"
        ;;
    *)
        echo "Error: Unknown backend '$BACKEND'. Use 'claude' or 'codex'."
        echo ""
        usage
        ;;
esac

# Check acpx is available
if ! command -v acpx &>/dev/null; then
    echo "Error: 'acpx' not found. Install it: npm install -g acpx"
    exit 1
fi

# Parse --no-slurm / --slurm flag
USE_SLURM=true
REMAINING_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-slurm)
            USE_SLURM=false
            ;;
        --slurm)
            USE_SLURM=true
            ;;
        *)
            REMAINING_ARGS+=("$arg")
            ;;
    esac
done
if [[ ${#REMAINING_ARGS[@]} -gt 0 ]]; then
    set -- "${REMAINING_ARGS[@]}"
else
    set --
fi

# Apply experiment mode override via temp config
if [[ "$USE_SLURM" == false ]]; then
    echo "[ResearchClaw] Experiments: local sandbox"
    TMP_CONFIG=$(mktemp "${SCRIPT_DIR}/.config_tmp_XXXXXX.yaml")
    sed 's/mode: "slurm"/mode: "sandbox"/' "$CONFIG" > "$TMP_CONFIG"
    CONFIG="$TMP_CONFIG"
    trap 'rm -f "$TMP_CONFIG"' EXIT
else
    echo "[ResearchClaw] Experiments: Slurm cluster"
fi

# Determine subcommand — if first arg starts with '-' or is absent, default to "run"
SUBCMD="${1:-run}"
if [[ "$SUBCMD" == -* ]]; then
    # First arg is a flag, not a subcommand — default to "run", keep all args
    SUBCMD="run"
else
    # First arg is a subcommand — consume it
    shift
fi

case "$SUBCMD" in
    setup)
        researchclaw setup
        ;;
    doctor|validate)
        researchclaw "$SUBCMD" --config "$CONFIG" "$@"
        ;;
    run)
        researchclaw run --config "$CONFIG" "$@"
        ;;
    *)
        researchclaw "$SUBCMD" --config "$CONFIG" "$@"
        ;;
esac
