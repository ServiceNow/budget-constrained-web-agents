#!/bin/bash

# Get the repository root directory (parent of scripts/)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -n "${1:-}" ]; then
    # Optional path given: look for log under {path}, run with that path
    BASE_PATH="$1"
    LOG_SEARCH_DIR="$BASE_PATH"
    WEBARENA_DIR="$BASE_PATH"
    RESULTS_SUBDIR="${2:-results}"
    RESULTS_DIR="$BASE_PATH/$RESULTS_SUBDIR"
else
    # No arg: current behavior — log in repo root, results under constbudg
    LOG_SEARCH_DIR="$REPO_ROOT"
    WEBARENA_DIR="$REPO_ROOT/constbudg"
    RESULTS_SUBDIR="results"
    RESULTS_DIR="$REPO_ROOT/constbudg/$RESULTS_SUBDIR"
fi

# Find all .log files in the log search directory
LOG_FILES=("$LOG_SEARCH_DIR"/*.log)

# Count the number of .log files
LOG_COUNT=0
for log_file in "${LOG_FILES[@]}"; do
    if [ -f "$log_file" ]; then
        ((LOG_COUNT++))
    fi
done

# Check if there is exactly one .log file
if [ "$LOG_COUNT" -eq 0 ]; then
    echo "Error: No .log files found in $LOG_SEARCH_DIR" >&2
    exit 1
elif [ "$LOG_COUNT" -gt 1 ]; then
    echo "Error: More than one .log file found in $LOG_SEARCH_DIR" >&2
    exit 1
fi

# Get the single .log file
LOG_FILE=""
for log_file in "${LOG_FILES[@]}"; do
    if [ -f "$log_file" ]; then
        LOG_FILE="$log_file"
        break
    fi
done

python "$REPO_ROOT/scripts/analyze_log_tokens.py" "$LOG_FILE"
echo -e "\n\n"
python "$REPO_ROOT/scripts/summarize_results.py" "$WEBARENA_DIR" --results-dir "$RESULTS_SUBDIR"
