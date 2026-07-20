"""Run-scoped paths: nested layout under constbudg/ (always includes a run id, default: "default")."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

_RUN_SUFFIX_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Single run id used when --run_suffix is omitted (and for path layout everywhere).
DEFAULT_RUN_SUFFIX = "default"

# WebArena site action/workflow basenames.
_WEBSITE_NAMES = ("admin", "gitlab", "map", "reddit", "shopping")


def sanitize_run_suffix(s: str | None) -> str:
    if s is None or not str(s).strip():
        return DEFAULT_RUN_SUFFIX
    s = str(s).strip()
    if not _RUN_SUFFIX_RE.fullmatch(s):
        raise ValueError(
            f"Invalid --run_suffix {s!r}: use only letters, digits, underscore, hyphen."
        )
    return s


def results_dir_for_suffix(suffix: str) -> str:
    return os.path.join("results", suffix)


def run_segment_from_config_dir(config_dir: str) -> str:
    """
    Segment after config_files/ (e.g. autoeval/log/<segment>/), or DEFAULT_RUN_SUFFIX if absent.
    """
    norm = os.path.normpath(config_dir)
    parts = norm.split(os.sep)
    try:
        i = next(j for j, p in enumerate(parts) if p == "config_files")
    except StopIteration:
        return DEFAULT_RUN_SUFFIX
    if i + 1 < len(parts) and parts[i + 1]:
        return parts[i + 1]
    return DEFAULT_RUN_SUFFIX


def ensure_config_dir_for_run(suffix: str) -> str:
    """Ensure config_files/<suffix>/ exists. Task JSON under .../<suffix>/<benchmark>/ comes from generate_test_data.py (not copied here)."""
    dest = Path("config_files") / suffix
    dest.mkdir(parents=True, exist_ok=True)
    return str(dest)


def ensure_action_tree_for_run(suffix: str) -> None:
    """Create actions/<suffix>/; seed missing files from flat actions/<site>.py if present, else clean.py."""
    dest_root = Path("actions") / suffix
    ensure_python_package_dir(dest_root)
    clean = Path("actions/clean.py")
    for name in _WEBSITE_NAMES:
        dest = dest_root / f"{name}.py"
        if dest.exists():
            continue
        flat = Path("actions") / f"{name}.py"
        if flat.is_file():
            shutil.copy2(flat, dest)
        elif clean.is_file():
            shutil.copy2(clean, dest)
        else:
            dest.write_text("")


def action_py_path(website: str, suffix: str) -> str:
    return os.path.join("actions", suffix, f"{website}.py")


def workflow_txt_path(website: str, suffix: str) -> str:
    return os.path.join("workflows", suffix, f"{website}.txt")


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_workflow_tree_for_run(suffix: str) -> None:
    """Create workflows/<suffix>/; seed missing files from flat workflows/<site>.txt if present, else clean.txt."""
    dest_root = Path("workflows") / suffix
    dest_root.mkdir(parents=True, exist_ok=True)
    clean = Path("workflows/clean.txt")
    for name in _WEBSITE_NAMES:
        dest = dest_root / f"{name}.txt"
        if dest.exists():
            continue
        flat = Path("workflows") / f"{name}.txt"
        if flat.is_file():
            shutil.copy2(flat, dest)
        elif clean.is_file():
            shutil.copy2(clean, dest)
        else:
            dest.write_text("")


def ensure_python_package_dir(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    init_py = dir_path / "__init__.py"
    if not init_py.exists():
        init_py.write_text("")


def resolve_config_dir_for_run_suffix(config_dir: str | None, run_suffix: str) -> str:
    """
    Align induce CLI with evaluate_trajectory: --config_dir is the task-config root
    ``config_files/<segment>`` (i.e. contains <benchmark>/<id>.json).

    When the user leaves the generic default (``config_files`` or ``config_files/default``),
    replace with ``config_files/<run_suffix>`` so subprocesses match ``ensure_config_dir_for_run``
    and autoeval's ``run_segment_from_config_dir``. Explicit paths are unchanged.
    """
    if not config_dir or not str(config_dir).strip():
        return os.path.join("config_files", run_suffix)
    norm = os.path.normpath(config_dir)
    generic = os.path.normpath("config_files")
    generic_default = os.path.normpath("config_files/default")
    if norm == generic or norm == generic_default:
        return os.path.join("config_files", run_suffix)
    return config_dir


def apply_induce_actions_run_suffix_paths(args) -> None:
    """Resolve nested paths from CONSTBUDG_RUN_SUFFIX (mutates args)."""
    run_suffix = os.environ.get("CONSTBUDG_RUN_SUFFIX", "").strip() or DEFAULT_RUN_SUFFIX
    setattr(args, "run_suffix", run_suffix)
    args.config_dir = resolve_config_dir_for_run_suffix(getattr(args, "config_dir", None), run_suffix)
    args.test_query_path = os.path.join("induce", "prompt", run_suffix, "test_query.txt")
    args.write_tests_dir = os.path.join("induce", "debug_actions", run_suffix)
    default_flat = os.path.join("actions", f"{args.website}.py")
    wap = getattr(args, "write_action_path", None)
    if wap is None or os.path.normpath(wap) == os.path.normpath(default_flat):
        args.write_action_path = os.path.join("actions", run_suffix, f"{args.website}.py")
    ensure_parent_dir(args.test_query_path)
    ensure_python_package_dir(Path(args.write_action_path).parent)


def apply_induce_memory_run_suffix_paths(args) -> None:
    """Resolve nested paths for induce_memory from CONSTBUDG_RUN_SUFFIX (mutates args)."""
    run_suffix = os.environ.get("CONSTBUDG_RUN_SUFFIX", "").strip() or DEFAULT_RUN_SUFFIX
    setattr(args, "run_suffix", run_suffix)
    args.config_dir = resolve_config_dir_for_run_suffix(getattr(args, "config_dir", None), run_suffix)
    args.test_query_path = os.path.join("induce", "prompt", run_suffix, "test_query.txt")
    args.write_tests_dir = os.path.join("induce", "debug_memory", run_suffix)
    default_flat = os.path.join("workflows", f"{args.website}.txt")
    wwp = getattr(args, "write_workflow_path", None)
    if wwp is None or os.path.normpath(wwp) == os.path.normpath(default_flat):
        args.write_workflow_path = os.path.join("workflows", run_suffix, f"{args.website}.txt")
    ensure_parent_dir(args.test_query_path)
    ensure_parent_dir(args.write_workflow_path)
