import os
import json
import argparse

from utils.unraisable import install_multiprocess_resource_tracker_noise_filter
from utils.procedures import evaluate_trajectory, calc_valid_steps, induce, demo_task_retry_on_timeout_early_or_playwright_error
from utils.tasks import get_task_names
from utils.run_context import (
    sanitize_run_suffix,
    results_dir_for_suffix,
    ensure_config_dir_for_run,
    workflow_txt_path,
    ensure_workflow_tree_for_run,
    ensure_action_tree_for_run,
)


def get_common_demo_args(task_name: str):
    return {
        "task_name": task_name,
        "max_steps": args.max_steps,
        "model_name": args.model_name,
        "prune_axtree": args.prune_axtree,
        "save_step_axtrees": args.save_step_axtrees,
        "results_dir": args.results_dir,
    }

# %% Baseline
def run_vanilla():
    for task_name in args.task_names:
        demo_task_retry_on_timeout_early_or_playwright_error(
            **get_common_demo_args(task_name),
            rename_to=task_name,
        )


# %% AWM
def run_awm():
    for task_name in args.task_names:
        # step 1: task solving
        if not demo_task_retry_on_timeout_early_or_playwright_error(
            **get_common_demo_args(task_name),
            rename_to=task_name,
            memory_path=args.workflow_txt_path,
        ): continue

        # # step 2: eval traj
        is_correct = evaluate_trajectory(
            task_name,
            model_name=args.eval_model_name,
            results_dir=args.results_dir,
            config_dir=args.config_dir,
        )
        if not is_correct: 
            print(f"Task {task_name} is judged as incorrect.")
            continue
        else:
            print(f"Task {task_name} is judged as correct, inducing workflow...")

        # step 3: induce workflows
        # step 3.1: clean trajectory
        print("\nCleaning trajectory...")
        calc_valid_steps(task_name, args.results_dir, model_name=args.model_name)

        # step 3.2: induce memory
        induce(
            "memory",
            args.benchmark,
            args.website,
            task_name.split(".")[-1],
            model_name=args.induction_model_name,
            results_dir=args.results_dir,
            config_dir=args.config_dir,
        )


# %% ASI
def run_asi():
    for task_name in args.task_names:
        # step 1: task solving
        if not demo_task_retry_on_timeout_early_or_playwright_error(
            **get_common_demo_args(task_name),
            rename_to=task_name,
            websites=args.website,
        ): continue

        path = f"{args.results_dir}/{task_name}/summary_info.json"
        if not os.path.isfile(path) or json.load(open(path, 'r')).get("n_steps", 0) < 3:
            continue

        # step 2: eval traj
        is_correct = evaluate_trajectory(
            task_name,
            model_name=args.eval_model_name,
            results_dir=args.results_dir,
            config_dir=args.config_dir,
        )
        if not is_correct: 
            print(f"Task {task_name} is judged as incorrect.")
            continue
        else:
            print(f"Task {task_name} is judged as correct, inducing actions...")
        
        # step 3.1: clean trajectory, output 'clean_steps.json'
        print("\nCleaning trajectory...")
        calc_valid_steps(task_name, args.results_dir, model_name=args.model_name)

        # step 3.2: induce actions
        induce(
            "actions",
            args.benchmark,
            args.website,
            task_name.split(".")[-1],
            model_name=args.induction_model_name,
            eval_model_name=args.eval_model_name,
            results_dir=args.results_dir,
            config_dir=args.config_dir,
        )




# %% Main Pipeline

if __name__ == "__main__":
    install_multiprocess_resource_tracker_noise_filter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["vanilla", "awm", "asi"])
    parser.add_argument("--benchmark", type=str, default="webarena")
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "admin", "reddit", "gitlab", "map"])
    parser.add_argument("--task_ids", type=str, default="0-811",
                        help="xxx-xxx,xxx-xxx")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="Maximum number of steps for the agent.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name to use (e.g. openrouter/google/gemini-3-flash-preview).")
    parser.add_argument("--eval_model_name", type=str, default="openrouter/openai/gpt-4o",
                        help="Model name to use for evaluation.")
    parser.add_argument("--induction_model_name", type=str, default=None,
                        help="Model name to use for induction.")
    parser.add_argument("--prune_axtree", action="store_true", default=False,
                        help="Prune the axtree before sending to the LLM (actor agent).")
    parser.add_argument("--save_step_axtrees", action="store_true", default=False,
                        help="Save raw and pruned accessibility trees for each observed step.")
    parser.add_argument("--run_suffix", type=str, default="default",
                        help="Run id; isolates results/<suffix>, config_files/<suffix>, workflows/<suffix>, ... (default: default).")

    args = parser.parse_args()
    args.induction_model_name = args.induction_model_name or args.model_name
    args.task_names = get_task_names(args.benchmark, args.website, args.task_ids)

    args.run_suffix = sanitize_run_suffix(args.run_suffix)
    args.results_dir = results_dir_for_suffix(args.run_suffix)
    args.config_dir = ensure_config_dir_for_run(args.run_suffix)
    args.workflow_txt_path = workflow_txt_path(args.website, args.run_suffix)
    os.environ["CONSTBUDG_RUN_SUFFIX"] = args.run_suffix
    os.environ["CONSTBUDG_CONFIG_DIR"] = args.config_dir

    ensure_action_tree_for_run(args.run_suffix)
    ensure_workflow_tree_for_run(args.run_suffix)

    if args.experiment == "vanilla":
        run_vanilla()
    elif args.experiment == "awm":
        run_awm()
    elif args.experiment == "asi":
        run_asi()
