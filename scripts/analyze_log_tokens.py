#!/usr/bin/env python3
"""
Script to analyze log files and extract LLM token usage statistics.
Finds lines matching: [<MODULE>] LLM: <LLM_NAME> | Tokens: <NUM> (prompt: <NUM>, completion: <NUM>)

By default, if a task subprocess was retried after timeout/error (see
``Retrying task … (attempt k/m)`` and ``Running experiment … _on_<task>_<seed>``),
only token lines from the final attempt are counted; use
``--include-non-final-retry-attempts`` to aggregate the entire log like before.
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def normalize_module_name(module: str) -> str:
    """
    Normalize module name by removing 'for task ...' suffix.
    
    Examples:
        "Actor agent for task webarena.117" -> "Actor agent"
        "induce_actions for task 47" -> "induce_actions"
    """
    # Remove "for task ..." pattern (case-insensitive, flexible spacing)
    pattern = r'\s+for\s+task\s+.*$'
    normalized = re.sub(pattern, '', module, flags=re.IGNORECASE)
    return normalized.strip()


# Start of a browsergym subprocess for one task (run_demo.py / browsergym).
# Example: "Running experiment DemoAgentArgs_on_webarena.51_12 in:"
RUNNING_EXPERIMENT_RE = re.compile(
    r"Running experiment\s+\w+_on_(\S+\.\d+)_\d+\s+in:"
)
# Printed by demo_task_retry_on_timeout_early_or_playwright_error before a new attempt.
# Example: "Retrying task webarena.51 (attempt 2/3)..."
RETRY_TASK_RE = re.compile(
    r"Retrying task\s+(\S+)\s+\(attempt\s+(\d+)/(\d+)\)"
)


def compute_non_final_attempt_excluded_ranges(lines: list[str]) -> List[Tuple[int, int]]:
    """
    Line index ranges (inclusive) whose LLM token lines should be skipped when
    only_final_retry_attempt is True: each subprocess run that was followed by
    a retry (timeout/error) for the same task name.

    Uses Running experiment ... _on_<task>_<seed> ... and Retrying task <task> (attempt k/m).
    """
    ranges: List[Tuple[int, int]] = []
    # Last unmatched Running experiment line index per task (most recent run awaiting outcome)
    pending_run_start: Dict[str, int] = {}

    for i, line in enumerate(lines):
        m_run = RUNNING_EXPERIMENT_RE.search(line)
        if m_run:
            task = m_run.group(1).strip()
            pending_run_start[task] = i
            continue

        m_retry = RETRY_TASK_RE.search(line)
        if not m_retry:
            continue
        task = m_retry.group(1).strip()
        attempt = int(m_retry.group(2))
        if attempt < 2:
            continue
        start = pending_run_start.pop(task, None)
        if start is None:
            print(
                f"Warning: retry for task {task!r} at line {i + 1} with no prior "
                f"'Running experiment' line for that task; not excluding any lines.",
                file=sys.stderr,
            )
            continue
        # Exclude everything from that run up through the line before the retry banner.
        ranges.append((start, i - 1))

    return merge_inclusive_ranges(ranges)


def merge_inclusive_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged: List[Tuple[int, int]] = [sorted_ranges[0]]
    for a, b in sorted_ranges[1:]:
        last_a, last_b = merged[-1]
        if a <= last_b + 1:
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))
    return merged


def line_index_in_ranges(i: int, ranges: List[Tuple[int, int]]) -> bool:
    for a, b in ranges:
        if a <= i <= b:
            return True
    return False


def parse_log_line(line: str) -> Tuple[str, str, int, int, int] | None:
    """
    Parse a log line to extract module, LLM name, and token counts.
    
    Returns:
        Tuple of (module, llm_name, total_tokens, prompt_tokens, completion_tokens) or None
    """
    # Pattern: [<MODULE>] LLM: <LLM_NAME> | Tokens: <NUM> (prompt: <NUM>, completion: <NUM>)
    pattern = r'\[([^\]]+)\]\s+LLM:\s+([^|]+)\s+\|\s+Tokens:\s+(\d+)\s+\(prompt:\s+(\d+),\s+completion:\s+(\d+)\)'
    
    match = re.search(pattern, line)
    if match:
        module = match.group(1).strip()
        # Normalize module name by removing "for task ..." suffix
        module = normalize_module_name(module)
        llm_name = match.group(2).strip()
        total_tokens = int(match.group(3))
        prompt_tokens = int(match.group(4))
        completion_tokens = int(match.group(5))
        return (module, llm_name, total_tokens, prompt_tokens, completion_tokens)
    
    return None


def analyze_log_file(
    log_path: Path,
    *,
    only_final_retry_attempt: bool = True,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Analyze log file and collect statistics grouped by module and LLM.

    If only_final_retry_attempt is True (default), token lines that appear in a
    browsergym subprocess run that was discarded and retried after
    timeout/error (see utils.procedures.demo_task_retry_on_timeout_early_or_playwright_error)
    are not counted — only the final attempt's usage is included.

    Returns:
        Nested dict: {module: {llm: {'calls': int, 'total_tokens': int, 'prompt_tokens': int, 'completion_tokens': int}}}
    """
    stats = defaultdict(lambda: defaultdict(lambda: {
        'calls': 0,
        'total_tokens': 0,
        'prompt_tokens': 0,
        'completion_tokens': 0
    }))
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Analyzing log file: {log_path}", file=sys.stderr)

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    excluded_ranges: List[Tuple[int, int]] = []
    print("Timeout tasks' actor agent token usage is not counted in the stats.")
    if only_final_retry_attempt:
        excluded_ranges = compute_non_final_attempt_excluded_ranges(lines)
        if excluded_ranges:
            total_excluded = sum(b - a + 1 for a, b in excluded_ranges)
            print(
                f"Omitting token lines in {len(excluded_ranges)} non-final retry "
                f"segment(s) ({total_excluded} log lines). "
                f"Use --include-non-final-retry-attempts to count all lines.",
                file=sys.stderr,
            )

    for line_num, line in enumerate(lines):
        if only_final_retry_attempt and line_index_in_ranges(line_num, excluded_ranges):
            continue
        result = parse_log_line(line)
        if result:
            module, llm_name, total_tokens, prompt_tokens, completion_tokens = result

            stats[module][llm_name]['calls'] += 1
            stats[module][llm_name]['total_tokens'] += total_tokens
            stats[module][llm_name]['prompt_tokens'] += prompt_tokens
            stats[module][llm_name]['completion_tokens'] += completion_tokens
    
    return stats


def print_statistics(stats: Dict[str, Dict[str, Dict[str, int]]]):
    """
    Print statistics grouped by module, then by LLM.
    """
    # Sort modules alphabetically
    sorted_modules = sorted(stats.keys())
    
    for module in sorted_modules:
        print(f"\n{'='*80}")
        print(f"Module: {module}")
        print(f"{'='*80}")
        
        # Sort LLMs alphabetically within each module
        sorted_llms = sorted(stats[module].keys())
        
        for llm in sorted_llms:
            data = stats[module][llm]
            print(f"\n  LLM: {llm}")
            print(f"    Calls: {data['calls']} (actor steps are squashed in this stats)")
            print(f"    Total Tokens: {data['total_tokens']:,}")
            print(f"    Prompt Tokens: {data['prompt_tokens']:,}")
            print(f"    Completion Tokens: {data['completion_tokens']:,}")
        
        # Module totals
        if len(sorted_llms) > 1:
            module_total_calls = sum(data['calls'] for data in stats[module].values())
            module_total_tokens = sum(data['total_tokens'] for data in stats[module].values())
            module_total_prompt = sum(data['prompt_tokens'] for data in stats[module].values())
            module_total_completion = sum(data['completion_tokens'] for data in stats[module].values())
            
            print(f"\n  Module Totals:")
            print(f"    Total Calls: {module_total_calls} (actor steps are squashed in this stats)")
            print(f"    Total Tokens: {module_total_tokens:,}")
            print(f"    Total Prompt Tokens: {module_total_prompt:,}")
            print(f"    Total Completion Tokens: {module_total_completion:,}")
    
    # Overall totals
    print(f"\n{'='*80}")
    print("OVERALL TOTALS")
    print(f"{'='*80}")
    
    overall_calls = sum(
        sum(data['calls'] for data in module_data.values())
        for module_data in stats.values()
    )
    overall_tokens = sum(
        sum(data['total_tokens'] for data in module_data.values())
        for module_data in stats.values()
    )
    overall_prompt = sum(
        sum(data['prompt_tokens'] for data in module_data.values())
        for module_data in stats.values()
    )
    overall_completion = sum(
        sum(data['completion_tokens'] for data in module_data.values())
        for module_data in stats.values()
    )
    
    print(f"Total Calls: {overall_calls} (actor steps are squashed in this stats)")
    print(f"Total Tokens: {overall_tokens:,}")
    print(f"Total Prompt Tokens: {overall_prompt:,}")
    print(f"Total Completion Tokens: {overall_completion:,}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract LLM token usage from constbudg/browsergym-style logs."
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to the log file",
    )
    parser.add_argument(
        "--include-non-final-retry-attempts",
        action="store_true",
        help=(
            "Count tokens from every attempt, including subprocess runs that were "
            "retried after timeout or error. "
            "Default is to omit those and count only the final attempt per retried task."
        ),
    )
    args = parser.parse_args()

    only_final_retry_attempt = not args.include_non_final_retry_attempts

    stats = analyze_log_file(
        args.log_file,
        only_final_retry_attempt=only_final_retry_attempt,
    )
    
    if not stats:
        print("No matching log entries found.", file=sys.stderr)
        return
    
    print_statistics(stats)


if __name__ == "__main__":
    main()

