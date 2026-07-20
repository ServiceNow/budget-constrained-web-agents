import json
import re
import argparse
from pathlib import Path


# Get project root (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent


def get_path(relative_path: str) -> Path:
    """Get absolute path relative to project root."""
    return PROJECT_ROOT / relative_path


def main():
    """Main function to summarize results."""
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Directory containing the experiment (should have results/ subdirectory)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Name or path of results directory under experiment_dir (default: results)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="webarena",
        help="Benchmark name (default: webarena)"
    )
    args = parser.parse_args()
    
    benchmark = args.benchmark
    
    # Resolve experiment directory path
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.is_absolute():
        experiment_dir = PROJECT_ROOT / experiment_dir
    
    if not experiment_dir.exists():
        print(f"Experiment directory not found: {experiment_dir}")
        return

    results_dir_arg = Path(args.results_dir)
    results_path = experiment_dir / results_dir_arg if not results_dir_arg.is_absolute() else results_dir_arg
    
    if not results_path.exists():
        print(f"Results directory not found: {results_path}")
        return
    
    # Pattern to match benchmark.<NUM> folders
    benchmark_pattern = re.compile(rf'^{re.escape(benchmark)}\.(\d+)$')
    
    # Pattern to match timeout folders: "<any string>_benchmark.<NUM1>_<NUM2>"
    # NUM1 is the task number
    timeout_folder_pattern = re.compile(rf'.*_{re.escape(benchmark)}\.(\d+)_\d+$')
    
    # Find timeout tasks by looking for folders matching the timeout pattern
    timeout_tasks = {}  # Map task_num -> timeout_folder_name
    for item in results_path.iterdir():
        if not item.is_dir():
            continue
        folder_name = item.name
        match = timeout_folder_pattern.match(folder_name)
        if match:
            task_num = int(match.group(1))
            timeout_tasks[task_num] = folder_name
    
    results = []
    total_tasks = 0
    successful_tasks = 0
    failed_tasks = 0
    error_tasks = 0
    missing_summary = 0
    timeout_count = 0
    
    # First, add timeout tasks that don't have regular benchmark.XX folders
    for task_num, timeout_folder_name in timeout_tasks.items():
        regular_folder_name = f"{benchmark}.{task_num}"
        regular_folder_path = results_path / regular_folder_name
        # Only add if regular folder doesn't exist (if it exists, we'll process it below)
        if not regular_folder_path.exists():
            results.append((task_num, regular_folder_name, "timeout", None, None))
            timeout_count += 1
            total_tasks += 1
    
    # Traverse results directory
    for item in results_path.iterdir():
        if not item.is_dir():
            continue
        
        folder_name = item.name
        
        # Skip folders ending with _test or containing _error_
        if folder_name.endswith("_test") or "_error_" in folder_name or "_attempt_" in folder_name:
            continue
        
        # Check if it matches benchmark.<NUM> pattern and extract number
        match = benchmark_pattern.match(folder_name)
        if not match:
            continue
        
        task_num = int(match.group(1))
        total_tasks += 1
        
        summary_file = item / "summary_info.json"
        
        if not summary_file.exists():
            results.append((task_num, folder_name, "missing", None, None))
            missing_summary += 1
            continue
        
        # Read and parse summary_info.json
        try:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            cum_reward = summary_data.get("cum_reward", None)
            err_msg = summary_data.get("err_msg", None)
            
            # If err_msg is not None, mark as ERR and count only as error task
            if err_msg is not None:
                status = "ERR"
                error_tasks += 1
            # Task is successful only if cum_reward == 1 AND err_msg is None
            elif cum_reward == 1:
                status = "success"
                successful_tasks += 1
            else:
                status = "failed"
                failed_tasks += 1
            
            results.append((task_num, folder_name, status, cum_reward, err_msg))
            
        except (json.JSONDecodeError, KeyError) as e:
            results.append((task_num, folder_name, "error", None, None))
            error_tasks += 1
    
    # Sort results by task number
    results.sort(key=lambda x: x[0])
    
    # Print sorted results
    for task_num, folder_name, status, cum_reward, err_msg in results:
        if status == "missing":
            print(f"{folder_name}: missing summary_info.json")
        elif status == "error":
            print(f"{folder_name}: error reading summary_info.json")
        elif status == "timeout":
            print(f"{folder_name}:\ttimeout")
        elif status == "ERR":
            print(f"{folder_name}:\t{status}\t(err_msg: {err_msg.replace('\n', '    ')})")
        else:
            print(f"{folder_name}:\t{status}")
    
    # Print overall statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total tasks: {total_tasks}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {failed_tasks}")
    print(f"Error tasks: {error_tasks}")
    print(f"Timeout tasks: {timeout_count}")
    print(f"Missing summary: {missing_summary}")
    
    if total_tasks > 0:
        success_rate = (successful_tasks / total_tasks) * 100
        print(f"Success rate: {success_rate:.2f}%")


if __name__ == "__main__":
    main()

