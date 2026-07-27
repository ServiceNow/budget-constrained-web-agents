#!/bin/bash
#
# Simple launcher for the constbudg pipeline (constbudg/run_online.py).
#
# It assumes your WebArena sites are already hosted and their URLs are set in
# `.env` (copy `.env.example` to `.env` first). This script loads your environment,
# optionally (re)generates task configs and warms up the browser, runs `run_online.py`,
# tees everything to a log file, and (optionally) writes a stats summary.
#
# Usage:
#   ./run.sh <log_file> [run_online.py args...]
#
# Examples:
#   # ASI on shopping
#   ./run.sh asi_shopping.log --experiment asi --website shopping
#
#   # Vanilla-IB on shopping
#   ./run.sh vanilla_ib_shopping.log --experiment vanilla --website shopping \
#       --max_steps 15 --prune_axtree
#
# Optional environment variables:
#   CLEANUP=no|delete|move   Clean prior artifacts for this --run_suffix before the run.
#                            no (default) keeps them; delete removes them; move archives
#                            them to warehouse/<log-name>/ (see scripts/cleanup.py).
#   GENCONFIG=1|0            (Re)generate webarena task configs for this --run_suffix before
#                            the run (default: 1). Needed by the awm/asi pipelines; the
#                            vanilla baseline does not use task configs (set GENCONFIG=0 to skip).
#   WARMUP=1|0               Run a tiny throwaway run_demo.py first to warm up the browser
#                            and site connection before the real run (default: 1).
#   STATS=1|0                After a successful run, archive results to warehouse/<log-name>/
#                            and write stats.txt there via scripts/stats.sh (default: 1).

set -o pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <log_file> [run_online.py args...]" >&2
    echo "  Must include --website (shopping|admin|reddit|gitlab|map) among the run_online.py args." >&2
    echo "Example: $0 logs/asi_shopping.log --experiment asi --website shopping --task_ids 21-25" >&2
    exit 1
fi

LOG_FILE="$1"
shift  # remaining args are passed through to run_online.py

CLEANUP="${CLEANUP:-no}"
GENCONFIG="${GENCONFIG:-1}"
WARMUP="${WARMUP:-1}"
STATS="${STATS:-1}"

# Locate the repo and package directories relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$SCRIPT_DIR"
CONSTBUDG_DIR="$WORKSPACE_DIR/constbudg"
ENV_FILE="$WORKSPACE_DIR/.env"

# Make the log path absolute so it survives the cd into the package directory.
if [[ "$LOG_FILE" != /* ]]; then
    LOG_FILE="$(cd "$(dirname "$LOG_FILE")" 2>/dev/null && pwd)/$(basename "$LOG_FILE")"
fi
LOG_BASENAME="${LOG_FILE##*/}"
LOG_NAME="${LOG_BASENAME%.log}"
mkdir -p "$(dirname "$LOG_FILE")"

ALL_ARGS=("$@")

get_arg_value() {
    local key="$1" i=0
    while [ $i -lt ${#ALL_ARGS[@]} ]; do
        if [ "${ALL_ARGS[$i]}" = "$key" ]; then
            i=$((i + 1))
            [ $i -lt ${#ALL_ARGS[@]} ] && echo "${ALL_ARGS[$i]}"
            return
        fi
        i=$((i + 1))
    done
}

WEBSITE_DOMAIN="$(get_arg_value "--website")"
if [ -z "$WEBSITE_DOMAIN" ]; then
    echo "Error: --website is required (e.g. --website shopping)." >&2
    exit 1
fi

# run_online.py always uses a run id (default: "default"); results live under
# constbudg/results/<run_suffix>/. Recover it here so cleanup/stats can be scoped.
RUN_SUFFIX="default"
for i in "${!ALL_ARGS[@]}"; do
    if [ "${ALL_ARGS[$i]}" = "--run_suffix" ]; then
        next_i=$((i + 1))
        [ $next_i -lt ${#ALL_ARGS[@]} ] && RUN_SUFFIX="${ALL_ARGS[$next_i]}"
        break
    fi
    if [[ "${ALL_ARGS[$i]}" =~ ^--run_suffix=(.*) ]]; then
        RUN_SUFFIX="${BASH_REMATCH[1]}"
        break
    fi
done

# Load environment variables (LLM keys + WA_* site URLs) from .env.
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "Loaded environment variables from $ENV_FILE" |& tee "$LOG_FILE"
else
    echo "Warning: .env file not found at $ENV_FILE (see .env.example)" |& tee "$LOG_FILE"
fi

# Optionally clean prior artifacts for this run_suffix before starting.
case "$CLEANUP" in
    delete)
        echo "Cleaning prior artifacts for run_suffix=$RUN_SUFFIX (delete)..." |& tee -a "$LOG_FILE"
        python "$WORKSPACE_DIR/scripts/cleanup.py" --run_suffix "$RUN_SUFFIX" --delete || \
            echo "Warning: cleanup failed." |& tee -a "$LOG_FILE"
        ;;
    move|yes)
        echo "Archiving prior artifacts for run_suffix=$RUN_SUFFIX to warehouse/tmp..." |& tee -a "$LOG_FILE"
        python "$WORKSPACE_DIR/scripts/cleanup.py" --run_suffix "$RUN_SUFFIX" tmp || \
            echo "Warning: cleanup failed." |& tee -a "$LOG_FILE"
        ;;
    *)
        echo "Skipping pre-run cleanup (CLEANUP=$CLEANUP)." |& tee -a "$LOG_FILE"
        ;;
esac

# Optionally (re)generate webarena task configs for this run id. The awm/asi
# pipelines (auto-eval + induction) need them; the vanilla baseline does not.
if [ "$GENCONFIG" = "1" ]; then
    echo "Generating webarena task configs for run_suffix=$RUN_SUFFIX ..." |& tee -a "$LOG_FILE"
    ( cd "$CONSTBUDG_DIR" && python config_files/generate_test_data.py \
        --input "config_files/test.raw.json" \
        --output-dir "config_files/${RUN_SUFFIX}/webarena" ) |& tee -a "$LOG_FILE"
fi

# Optionally run a tiny throwaway run_demo.py to warm up the browser / site connection.
if [ "$WARMUP" = "1" ]; then
    DUMMY_MODEL=$(get_arg_value "--model_name")
    [ -z "$DUMMY_MODEL" ] && DUMMY_MODEL="openrouter/openai/gpt-4o"
    DUMMY_BENCHMARK=$(get_arg_value "--benchmark")
    [ -z "$DUMMY_BENCHMARK" ] && DUMMY_BENCHMARK="webarena"
    case "$WEBSITE_DOMAIN" in
        shopping) DUMMY_TID=21 ;;
        admin)    DUMMY_TID=0 ;;
        reddit)   DUMMY_TID=27 ;;
        gitlab)   DUMMY_TID=102 ;;
        map)      DUMMY_TID=378 ;;
        *)        DUMMY_TID=21 ;;
    esac
    DUMMY_TASK_NAME="${DUMMY_BENCHMARK}.${DUMMY_TID}"
    DUMMY_WARMUP_DIR=$(mktemp -d "${WORKSPACE_DIR}/.dummy_warmup.XXXXXX")
    echo "Warming up with run_demo.py (task ${DUMMY_TASK_NAME}, website ${WEBSITE_DOMAIN})..." |& tee -a "$LOG_FILE"
    ( cd "$CONSTBUDG_DIR" && python run_demo.py \
        --task_name "$DUMMY_TASK_NAME" \
        --model_name "$DUMMY_MODEL" \
        --results_dir "$DUMMY_WARMUP_DIR" \
        --max_steps 2 \
        --websites "$WEBSITE_DOMAIN" \
        --headless &>/dev/null ) || true
    rm -rf "$DUMMY_WARMUP_DIR"
    echo "Warmup finished (output ignored)." |& tee -a "$LOG_FILE"
fi

# Log the command that is about to run.
{
    echo "=========================================="
    echo "python run_online.py ${ALL_ARGS[*]}"
    echo "  Working directory: $CONSTBUDG_DIR"
    echo "  Python: $(command -v python)"
    echo "  Timestamp: $(date +"%Y-%m-%dT%T") ($(date))"
    echo "=========================================="
} |& tee -a "$LOG_FILE"

# Run the pipeline.
cd "$CONSTBUDG_DIR" || exit 1
python -u run_online.py "${ALL_ARGS[@]}" |& tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

{
    echo ""
    echo "=========================================="
    echo "Exit code: $EXIT_CODE  (finished at $(date))"
    echo "=========================================="
} |& tee -a "$LOG_FILE"

# On success, optionally archive this run to warehouse/<log-name>/ and write stats.
if [ "$EXIT_CODE" -eq 0 ] && [ "$STATS" = "1" ]; then
    cd "$SCRIPT_DIR" || exit 1
    WAREHOUSE_DIR="$WORKSPACE_DIR/warehouse/$LOG_NAME"
    echo "Archiving run to warehouse/${LOG_NAME} (run_suffix=${RUN_SUFFIX}) and writing stats..." |& tee -a "$LOG_FILE"
    if python "$WORKSPACE_DIR/scripts/cleanup.py" --run_suffix "$RUN_SUFFIX" --log_path "$LOG_FILE" "$LOG_NAME"; then
        mkdir -p "$WAREHOUSE_DIR"
        ./scripts/stats.sh "$WAREHOUSE_DIR" "results" > "$WAREHOUSE_DIR/stats.txt"
        echo "Stats written to warehouse/${LOG_NAME}/stats.txt" >&2
    else
        echo "Warning: archive step failed; stats not written." >&2
    fi
fi

exit $EXIT_CODE
