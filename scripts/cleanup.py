#!/usr/bin/env python3
"""Cleanup script to delete files and folders created by various procedures."""

import os
import shutil
import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
_CONSTBUDG = PROJECT_ROOT / "constbudg"
if str(_CONSTBUDG) not in sys.path:
    sys.path.insert(0, str(_CONSTBUDG))
from utils.run_context import DEFAULT_RUN_SUFFIX, sanitize_run_suffix


def _warehouse_rel(rel: Path, run_suffix: str) -> Path:
    """Path under warehouse/<name>/: drop leading run_suffix segment."""
    if not rel.parts or rel.parts[0] != run_suffix:
        return rel
    return Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(".")


def _move_tree_to_warehouse(src: Path, dest: Path) -> None:
    """Move src directory to dest (dest parent created; existing dest removed)."""
    if not src.is_dir():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        _rmtree_resilient(dest)
    print(f"Moving: {src} -> {dest}")
    shutil.move(str(src), str(dest))


def _move_flat_files_to_warehouse(src_dir: Path, dest_dir: Path, *glob_patterns: str) -> None:
    """Move each file under src_dir matching any glob into dest_dir (no suffix subfolder). Remove src_dir when done."""
    if not src_dir.is_dir():
        return
    if not glob_patterns:
        glob_patterns = ("*",)
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for pattern in glob_patterns:
        for f in sorted(src_dir.glob(pattern)):
            if not f.is_file() or f in seen:
                continue
            seen.add(f)
            dest = dest_dir / f.name
            if dest.exists():
                dest.unlink()
            print(f"Moving: {f} -> {dest}")
            shutil.move(str(f), str(dest))
    for cache in list(src_dir.rglob("__pycache__")):
        if cache.is_dir():
            _rmtree_resilient(cache)
    if src_dir.is_dir():
        _rmtree_resilient(src_dir)


def _rmtree_resilient(path: Path, max_retries: int = 3) -> bool:
    """Remove directory tree, retrying on OSError (e.g. directory not empty / busy)."""
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            return True
        except OSError as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                print(f"Warning: could not remove {path}: {e}")
                return False
    return False


def get_path(relative_path: str) -> Path:
    """Get absolute path relative to project root."""
    return PROJECT_ROOT / relative_path


def remove_config_files_suffix(run_suffix: str) -> None:
    """Remove constbudg/config_files/<run_suffix>/ (rebuilt by generate_test_data / run_online on next run)."""
    cfg = get_path(f"constbudg/config_files/{run_suffix}")
    if cfg.is_dir():
        print(f"Removing config_files run dir: {cfg}")
        _rmtree_resilient(cfg)


def _remove_empty_dirs_bottom_up(root: Path) -> None:
    """Remove empty directories under root, deepest first, then root if empty."""
    if not root.is_dir():
        return
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    dirs.sort(key=lambda p: len(p.parts), reverse=True)
    for d in dirs:
        try:
            if not any(d.iterdir()):
                d.rmdir()
                print(f"Removed empty directory: {d}")
        except OSError as e:
            print(f"Warning: could not remove directory {d}: {e}")
    try:
        if root.is_dir() and not any(root.iterdir()):
            root.rmdir()
            print(f"Removed empty directory: {root}")
    except OSError as e:
        print(f"Warning: could not remove directory {root}: {e}")


def remove_empty_suffix_scoped_dirs(run_suffix: str) -> None:
    """Remove empty trees under run-scoped dirs (results, outputs, induce debug_actions/prompt, etc.)."""
    for rel in (
        f"constbudg/results/{run_suffix}",
        f"constbudg/outputs/{run_suffix}",
        f"constbudg/induce/debug_actions/{run_suffix}",
        f"constbudg/induce/prompt/{run_suffix}",
    ):
        _remove_empty_dirs_bottom_up(get_path(rel))


def clean_run_demo(results_path: Path, warehouse_path: Path = None, run_suffix: str = DEFAULT_RUN_SUFFIX):
    """Delete all result directories created by run_demo.py under results_path."""
    if not results_path.exists():
        return

    for item in results_path.iterdir():
        if item.is_dir():
            if warehouse_path:
                dest = warehouse_path / "results" / item.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"Moving: {item} -> {dest}")
                shutil.move(str(item), str(dest))
            else:
                print(f"Removing: {item}")
                _rmtree_resilient(item)


def clean_induce_actions(results_path: Path, warehouse_path: Path = None, run_suffix: str = DEFAULT_RUN_SUFFIX):
    """Delete all files created by induce_actions.py and reset action files."""
    # Test files (flat and nested induce/debug_actions/<suffix>/)
    tests_root = get_path("constbudg/induce/debug_actions")
    if tests_root.exists():
        scope = tests_root / run_suffix
        test_files = list(scope.rglob("test_*.txt")) if scope.is_dir() else []
        for test_file in test_files:
            if warehouse_path:
                rel = _warehouse_rel(test_file.relative_to(tests_root), run_suffix)
                dest = warehouse_path / "induce" / "debug_actions" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"Moving: {test_file} -> {dest}")
                shutil.move(str(test_file), str(dest))
            else:
                print(f"Removing: {test_file}")
                test_file.unlink()

    # Output directories (action_task-*), including outputs/<suffix>/...
    outputs_path = get_path("constbudg/outputs")
    if outputs_path.exists():
        search_roots = [outputs_path / run_suffix] if (outputs_path / run_suffix).is_dir() else []
        for root in search_roots:
            for action_dir in root.rglob("action_task-*"):
                if not action_dir.is_dir():
                    continue
                if warehouse_path:
                    rel = _warehouse_rel(action_dir.relative_to(outputs_path), run_suffix)
                    dest = warehouse_path / "outputs" / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Moving: {action_dir} -> {dest}")
                    shutil.move(str(action_dir), str(dest))
                else:
                    print(f"Removing: {action_dir}")
                    _rmtree_resilient(action_dir)

    # Test query file(s)
    prompt_root = get_path("constbudg/induce/prompt")
    if prompt_root.exists():
        candidates = [prompt_root / run_suffix / "test_query.txt"]
        test_queries = [p for p in candidates if p.is_file()]
        for test_query in test_queries:
            if warehouse_path:
                rel = _warehouse_rel(test_query.relative_to(prompt_root), run_suffix)
                dest = warehouse_path / "induce" / "prompt" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"Moving: {test_query} -> {dest}")
                shutil.move(str(test_query), str(dest))
            else:
                print(f"Removing: {test_query}")
                test_query.unlink()

    # Test result directories (*_test)
    if results_path.exists():
        for item in results_path.iterdir():
            if item.is_dir() and item.name.endswith("_test"):
                if warehouse_path:
                    dest = warehouse_path / "results" / item.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Moving: {item} -> {dest}")
                    shutil.move(str(item), str(dest))
                else:
                    print(f"Removing: {item}")
                    _rmtree_resilient(item)

    actions_path = get_path("constbudg/actions")
    sub = actions_path / run_suffix
    if warehouse_path and sub.is_dir():
        # Flatten: warehouse/<wh>/actions/*.py (no actions/<suffix>/ subfolder)
        _move_flat_files_to_warehouse(sub, warehouse_path / "actions", "*.py", "*.py.tmp")
    elif actions_path.exists() and sub.is_dir():
        # Backup files (*.py.tmp) then reset each site file to clean.py
        tmp_iter = sub.rglob("*.py.tmp")
        for backup_file in tmp_iter:
            print(f"Removing: {backup_file}")
            backup_file.unlink()

        clean_file = get_path("constbudg/actions/clean.py")
        if not clean_file.exists():
            return

        clean_content = clean_file.read_text()
        clean_content_stripped = clean_content.strip()

        action_names = ["admin", "gitlab", "map", "reddit", "shopping"]

        def _reset_action_py(action_file: Path) -> None:
            if not action_file.exists():
                return
            current_content = action_file.read_text()
            if current_content.strip() == clean_content_stripped:
                return
            print(f"Resetting: {action_file}")
            action_file.write_text(clean_content)

        for name in action_names:
            _reset_action_py(sub / f"{name}.py")


def clean_induce_memory(warehouse_path: Path = None, run_suffix: str = DEFAULT_RUN_SUFFIX):
    """Delete all files created by induce_memory.py."""
    # Output directories (workflow_task-*), including nested outputs/<suffix>/...
    outputs_path = get_path("constbudg/outputs")
    if outputs_path.exists():
        search_roots = [outputs_path / run_suffix] if (outputs_path / run_suffix).is_dir() else []
        for root in search_roots:
            for workflow_dir in root.rglob("workflow_task-*"):
                if not workflow_dir.is_dir():
                    continue
                if warehouse_path:
                    rel = _warehouse_rel(workflow_dir.relative_to(outputs_path), run_suffix)
                    dest = warehouse_path / "outputs" / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Moving: {workflow_dir} -> {dest}")
                    shutil.move(str(workflow_dir), str(dest))
                else:
                    print(f"Removing: {workflow_dir}")
                    _rmtree_resilient(workflow_dir)

    # induce_memory shares test_query.txt handling with induce_actions; clean_induce_actions already moves prompt/**/test_query.txt


def clean_awm(results_path: Path, warehouse_path: Path = None, run_suffix: str = DEFAULT_RUN_SUFFIX):
    """Remove AWM debug artifacts and reset workflow .txt files to workflows/clean.txt (like actions/clean.py)."""
    debug_root = get_path("constbudg/induce/debug_memory")
    if debug_root.exists():
        scope = debug_root / run_suffix
        test_files = list(scope.rglob("test_*.txt")) if scope.is_dir() else []
        for test_file in test_files:
            if warehouse_path:
                rel = _warehouse_rel(test_file.relative_to(debug_root), run_suffix)
                dest = warehouse_path / "induce" / "debug_memory" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"Moving: {test_file} -> {dest}")
                shutil.move(str(test_file), str(dest))
            else:
                print(f"Removing: {test_file}")
                test_file.unlink()

    workflows_root = get_path("constbudg/workflows")
    wf_sub = workflows_root / run_suffix
    if not workflows_root.exists():
        return

    if warehouse_path and wf_sub.is_dir():
        _move_flat_files_to_warehouse(wf_sub, warehouse_path / "workflows", "*.txt")
        return

    clean_txt = workflows_root / "clean.txt"
    clean_content = clean_txt.read_text() if clean_txt.exists() else ""
    clean_stripped = clean_content.strip()

    def _reset_wf(wf: Path) -> None:
        if wf.name == "clean.txt":
            return
        cur = wf.read_text() if wf.exists() else ""
        if cur.strip() == clean_stripped:
            return
        print(f"Resetting workflow file to clean baseline: {wf}")
        wf.write_text(clean_content)

    if wf_sub.is_dir():
        for wf in wf_sub.glob("*.txt"):
            if wf.name == "clean.txt":
                continue
            _reset_wf(wf)


def clean_calc_valid_steps(results_path: Path, warehouse_path: Path = None, run_suffix: str = DEFAULT_RUN_SUFFIX):
    """Delete all cleaned_steps.json files created by calc_valid_steps.py."""
    if not results_path.exists():
        return
    
    for item in results_path.iterdir():
        if item.is_dir():
            cleaned_file = item / "cleaned_steps.json"
            if cleaned_file.exists():
                if warehouse_path:
                    dest = warehouse_path / "results" / item.name / cleaned_file.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Moving: {cleaned_file} -> {dest}")
                    shutil.move(str(cleaned_file), str(dest))
                else:
                    print(f"Removing: {cleaned_file}")
                    cleaned_file.unlink()


def clean_autoeval(results_path: Path, warehouse_path: Path = None, run_suffix: str = DEFAULT_RUN_SUFFIX):
    """Delete all files created by autoeval."""
    # Autoeval log directories (flat autoeval/log/<task> or nested autoeval/log/<suffix>/<task>)
    log_path = get_path("constbudg/autoeval/log")
    if log_path.exists():
        scoped_log = log_path / run_suffix
        items = (
            [x for x in scoped_log.iterdir() if x.is_dir()]
            if scoped_log.is_dir()
            else []
        )
        for item in items:
            if warehouse_path:
                dest = warehouse_path / "autoeval" / "log" / item.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"Moving: {item} -> {dest}")
                shutil.move(str(item), str(dest))
            else:
                print(f"Removing: {item}")
                _rmtree_resilient(item)
        if scoped_log.is_dir():
            try:
                if not any(scoped_log.iterdir()):
                    scoped_log.rmdir()
            except OSError:
                pass

    # Autoeval JSON files in result directories
    if results_path.exists():
        for item in results_path.iterdir():
            if item.is_dir():
                for autoeval_file in item.glob("*_autoeval.json"):
                    if warehouse_path:
                        dest = warehouse_path / "results" / item.name / autoeval_file.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        print(f"Moving: {autoeval_file} -> {dest}")
                        shutil.move(str(autoeval_file), str(dest))
                    else:
                        print(f"Removing: {autoeval_file}")
                        autoeval_file.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Clean up files created by procedures. Default: move artifacts to warehouse/<name>. "
        "Use --delete to remove instead."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete artifacts instead of moving to warehouse (no warehouse_name).",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Log file to move into warehouse when archiving (optional).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Unused; cleanup always uses constbudg/results/<run_suffix> (default suffix: default).",
    )
    parser.add_argument(
        "--run_suffix",
        type=str,
        default="default",
        help="Run id to clean; scopes results/<suffix>, outputs/<suffix>, workflows/<suffix>, ... (default: default).",
    )
    parser.add_argument(
        "--procedure",
        "-p",
        type=str,
        default="all",
        choices=[
            "run_demo",
            "induce_actions",
            "induce_memory",
            "awm",
            "calc_valid_steps",
            "autoeval",
            "all",
        ],
        help="Which cleanup steps to run for this run_suffix (default: all).",
    )
    parser.add_argument(
        "warehouse_name",
        nargs="?",
        default=None,
        help="Subdirectory under warehouse/ (required unless --delete).",
    )

    args = parser.parse_args()

    if args.delete and args.warehouse_name:
        parser.error("Do not pass warehouse_name when using --delete")
    if not args.delete and (not args.warehouse_name or not str(args.warehouse_name).strip()):
        parser.error("warehouse_name is required unless --delete")

    try:
        run_suffix = sanitize_run_suffix(args.run_suffix)
    except ValueError as e:
        parser.error(str(e))
    results_path = PROJECT_ROOT / "constbudg" / "results" / run_suffix
    print(f"Cleanup scoped to --run_suffix={run_suffix!r} (results: {results_path})")

    warehouse_path = None
    if not args.delete:
        wn = args.warehouse_name.strip()
        warehouse_path = get_path(f"warehouse/{wn}")
        warehouse_path.mkdir(parents=True, exist_ok=True)
        print(f"Moving files to: {warehouse_path}")
        
        # Move log files to warehouse
        if args.log_path:
            # Move specific log file/directory if log_path is set
            log_path = Path(args.log_path)
            # Handle both absolute and relative paths
            if not log_path.is_absolute():
                log_path = PROJECT_ROOT / log_path
            
            if log_path.exists():
                if log_path.is_file():
                    dest = warehouse_path / log_path.name
                    print(f"Moving log file: {log_path} -> {dest}")
                    shutil.move(str(log_path), str(dest))
                    # Move stats.txt from same location if present
                    stats_file = log_path.parent / "stats.txt"
                    if stats_file.is_file():
                        stats_dest = warehouse_path / "stats.txt"
                        print(f"Moving stats file: {stats_file} -> {stats_dest}")
                        shutil.move(str(stats_file), str(stats_dest))
                elif log_path.is_dir():
                    dest = warehouse_path / log_path.name
                    print(f"Moving log directory: {log_path} -> {dest}")
                    shutil.move(str(log_path), str(dest))
                    # Move stats.txt from same location (parent dir) if present
                    stats_file = log_path.parent / "stats.txt"
                    if stats_file.is_file():
                        stats_dest = warehouse_path / "stats.txt"
                        print(f"Moving stats file: {stats_file} -> {stats_dest}")
                        shutil.move(str(stats_file), str(stats_dest))
                else:
                    print(f"Warning: Log path exists but is not a file or directory: {log_path}")
            else:
                print(f"Warning: Log path does not exist: {log_path}")
        else:
            # If log_path is not set: prefer a single repo-root log named {warehouse_name}.log if it exists
            named_log = PROJECT_ROOT / f"{wn}.log"
            if named_log.is_file():
                dest = warehouse_path / named_log.name
                print(f"Moving log file: {named_log} -> {dest}")
                shutil.move(str(named_log), str(dest))
            else:
                # Move all *.log files from repo root
                print(
                    f"Warning: archiving to warehouse/{wn!r}, but --log_path is not set and no {wn}.log at repo root. "
                    "Moving *.log files from repo root to warehouse."
                )
                for log_file in PROJECT_ROOT.glob("*.log"):
                    if log_file.is_file():
                        dest = warehouse_path / log_file.name
                        print(f"Moving log file: {log_file} -> {dest}")
                        shutil.move(str(log_file), str(dest))
                # Move stats.txt from repo root if present (same location as *.log files)
                stats_file = PROJECT_ROOT / "stats.txt"
                if stats_file.is_file():
                    stats_dest = warehouse_path / "stats.txt"
                    print(f"Moving stats file: {stats_file} -> {stats_dest}")
                    shutil.move(str(stats_file), str(stats_dest))
    
    procedures = {
        "run_demo": lambda wh: clean_run_demo(results_path, wh, run_suffix),
        "induce_actions": lambda wh: clean_induce_actions(results_path, wh, run_suffix),
        "induce_memory": lambda wh: clean_induce_memory(wh, run_suffix),
        "awm": lambda wh: clean_awm(results_path, wh, run_suffix),
        "calc_valid_steps": lambda wh: clean_calc_valid_steps(results_path, wh, run_suffix),
        "autoeval": lambda wh: clean_autoeval(results_path, wh, run_suffix),
    }

    if args.procedure == "all":
        for key in (
            "run_demo",
            "induce_actions",
            "induce_memory",
            "awm",
            "calc_valid_steps",
            "autoeval",
        ):
            procedures[key](warehouse_path)
    else:
        procedures[args.procedure](warehouse_path)

    remove_config_files_suffix(run_suffix)
    remove_empty_suffix_scoped_dirs(run_suffix)


if __name__ == "__main__":
    main()
