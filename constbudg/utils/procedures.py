import json
import subprocess
import os
import shutil
import tempfile
import time
from pathlib import Path

_PLAYWRIGHT_IMPL_ERRORS_PREFIX = "playwright._impl._errors."


def _subprocess_env() -> dict:
    """Child processes inherit the current environment (including CONSTBUDG_RUN_SUFFIX)."""
    return os.environ.copy()


def run_demo_task(task_name: str, results_dir: str, timeout: int = 1000, **kwargs) -> bool:
    """
    Run run_demo.py with the given parameters and handle timeout.
    
    Returns:
        bool: True if process completed successfully, False if timed out
    """
    cmd = ["python", "run_demo.py", "--task_name", task_name, "--results_dir", results_dir]
    
    kwargs["headless"] = kwargs.get("headless", True)
    for key, value in kwargs.items():
        if not value:
            continue
        if isinstance(value, bool):
            cmd.extend([f"--{key}"])
        else:
            cmd.extend([f"--{key}", str(value)])
    
    process = subprocess.Popen(cmd, env=_subprocess_env())
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        print("Process completed successfully:")
        print(stdout)
        return True
    except subprocess.TimeoutExpired as e:
        process.kill()
        stdout, stderr = process.communicate()  # Clean up resources
        print(f"Process timed out after {e.timeout} seconds for task {task_name}.")
        print(stderr)
        return False

def evaluate_trajectory(
    task_name: str,
    results_dir: str,
    model_name: str = "openrouter/openai/gpt-4o",
    prompt: str = None,
    config_dir: str = "config_files/default",
) -> tuple[bool, str | None]:
    """
    Evaluate a trajectory and optionally generate hints for failed tasks.
    
    Args:
        task_name: Name of the task to evaluate.
        model_name: Model to use for evaluation.
        results_dir: Base directory for results.
        prompt: Prompt version ('text' or 'vision').
    
    Returns:
        bool: is_correct.
    """
    cmd = [
        "python", "-m", "autoeval.evaluate_trajectory",
        "--result_dir", f"{results_dir}/{task_name}",
        "--model", model_name,
        "--config_dir", config_dir,
    ]
    if prompt is not None:
        cmd.extend(["--prompt", prompt])

    process = subprocess.Popen(cmd, env=_subprocess_env())
    process.wait()
    actual_model_name = model_name.split("/")[-1]
    try:
        eval_result = json.load(open(f"{results_dir}/{task_name}/{actual_model_name}_autoeval.json"))[0]
    except FileNotFoundError:
        print(f"File {results_dir}/{task_name}/{actual_model_name}_autoeval.json not found.")
        return False
    is_correct = eval_result["rm"]  # bool
    return is_correct

def calc_valid_steps(
    task_name: str,
    results_dir: str,
    model_name: str = None,
    clean_and_store: bool = True,
    action_names: list[str] = None,
    timeout: int = 900,
    num_retries: int = 1,
) -> str:
    cmd = [
        "python", "-m", "utils.calc_valid_steps",
        "--result_dir", f"{results_dir}/{task_name}",
    ]
    if clean_and_store:
        cmd.append("--clean_and_store")
    if model_name is not None:
        cmd.extend(["--model", model_name])
    if action_names is not None:
        cmd.extend(["--action_names"] + action_names)

    for attempt in range(1, num_retries + 2):
        if attempt > 1:
            print(
                f"Retrying calc_valid_steps for {task_name} "
                f"(attempt {attempt}/{num_retries + 1})..."
            )
            time.sleep(60)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_subprocess_env(),
        )
        try:
            stdout, _ = process.communicate(timeout=timeout)
            if stdout:
                print(stdout, end="")
            return stdout.strip()
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _ = process.communicate()
            print(f"calc_valid_steps timed out after {timeout} seconds for {task_name}.")
            if stdout:
                print(stdout, end="")
            if attempt == num_retries + 1:
                return "False"

    return "False"

def induce(
    induce_type: str,
    benchmark: str,
    website: str,
    result_id_list: str,
    results_dir: str,
    model_name: str = None,
    eval_model_name: str = None,
    timeout: int = 600,
    config_dir: str = "config_files/default",
) -> bool:
    """
    Induce either actions or memory based on the type.
    """
    if induce_type not in ["actions", "memory"]:
        raise ValueError(f"induce_type must be 'actions' or 'memory', got '{induce_type}'")
    
    cmd = [
        "python", "-m", f"induce.induce_{induce_type}",
        "--benchmark", benchmark,
        "--website", website,
        "--result_id_list", result_id_list,
        "--results_dir", results_dir,
        "--config_dir", config_dir,
    ]
    if model_name is not None:
        cmd.extend(["--model", model_name])
    if eval_model_name is not None:
        cmd.extend(["--eval_model_name", eval_model_name])
    process = subprocess.Popen(cmd, env=_subprocess_env())
    if timeout is not None:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            if stdout:
                print(stdout)
            return True
        except subprocess.TimeoutExpired as e:
            process.kill()
            stdout, stderr = process.communicate()  # Clean up resources
            task_names = " & ".join([f"{benchmark}.{t}" for t in result_id_list.split(",")])
            print(f"Process timed out after {e.timeout} seconds for task {task_names}.")
            if stderr:
                print(stderr)
            return False
    else:
        process.wait()
        return True

def demo_task_retry_on_timeout_early_or_playwright_error(task_name: str, rename_to: str, results_dir: str, num_retries: int = 2, **kwargs) -> bool:
    """
    Run run_demo_task and retry in these conditions (based on summary_info.json): 
        - no steps completed and any error is detected,
        - any playwright._impl._errors.* (e.g. TargetClosedError), or
        - timeout is detected
    
    Args:
        task_name: Name of the task to run
        rename_to: Rename the output folder to this name
        num_retries: Maximum number of retries after the first run
        results_dir: Base directory for results
    
    Returns:
        bool: Retruns the final run_demo_task call's output
    """
    def _should_retry() -> bool:
        summary_path = os.path.join(results_dir, rename_to, "summary_info.json")
        if not os.path.exists(summary_path):
            return False
        
        try:
            with open(summary_path, 'r') as f:
                summary_info = json.load(f)
            
            n_steps = summary_info.get("n_steps", 0)
            err_msg = summary_info.get("err_msg") or ""
            stack_trace = summary_info.get("stack_trace") or ""
            if n_steps == 0 and (err_msg or stack_trace):  # no steps completed, probably due to browsergym connection issue
                return True

            if _PLAYWRIGHT_IMPL_ERRORS_PREFIX in (err_msg + "\n" + stack_trace):  # Playwright impl error
                return True
        except (json.JSONDecodeError, KeyError, IOError):
            pass
        return False

    def _rename_folder(index: int):
        old_folder_path = Path(results_dir, rename_to)
        new_folder_name = f"{rename_to}_error_{index}"
        new_folder_path = old_folder_path.parent / new_folder_name
        if old_folder_path.exists() and not new_folder_path.exists():
            print(f"Renaming folder: {rename_to} -> {new_folder_name}")
            shutil.move(str(old_folder_path), str(new_folder_path))
    

    for attempt in range(1, num_retries + 2):
        if attempt > 1:
            print(f"Retrying task {task_name} (attempt {attempt}/{num_retries + 1})...")
            _rename_folder(attempt - 1)
        
        done = run_demo_task(task_name=task_name, rename_to=rename_to, results_dir=results_dir, **kwargs)
        
        if done and not _should_retry():
            return done
        else:
            print(f"Detected {'error' if done else 'timeout'}. Waiting for 1 minute before retrying...")
            time.sleep(60)
    return done
