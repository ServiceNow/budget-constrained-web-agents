"""Induce Actions on the Full Task Level."""

import os
import gzip
import json
import pickle
import argparse
import subprocess
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_client import llm_completion, get_contents
from utils.procedures import (
    run_demo_task,
    evaluate_trajectory,
    calc_valid_steps,
)
from induce.utils import extract_code_pieces, get_task_id, get_result_dirs, get_output_dir
from utils.run_context import apply_induce_actions_run_suffix_paths

# %% Induce Actions

def get_example_query_filtered(index: int, result_dir: str, config_dir: str, benchmark: str = "webarena") -> tuple[str, str]:
    """Get the string for a past example experience, without invalid actions."""
    cid = get_task_id(result_dir)
    config_path = os.path.join(config_dir, benchmark, f"{cid}.json")
    config = json.load(open(config_path))
    instruction = config["intent"]
    task = config["intent_template"]

    step_dirs = [f for f in os.listdir(result_dir) if f.startswith("step") and f.endswith(".pkl.gz")]
    step_dirs = sorted(step_dirs, key=lambda x: int(x.split('.')[0].split('_')[1]))
    step_dirs = [os.path.join(result_dir, sd) for sd in step_dirs]
    steps = []
    for sd in step_dirs:
        step_info = pickle.load(gzip.open(sd, 'rb'))
        error_msg = step_info.obs["last_action_error"]
        if len(error_msg) > 0 and (not error_msg.startswith("TimeoutError")):
            continue  # skip error actions
        if len(step_info.obs["last_action"]) == 0:
            continue  # skip empty actions
        steps.append(step_info.obs["last_action"])

    if len(steps) > 0:
        print(f"Collected #{len(steps)} valid steps from a total of #{len(step_dirs)} steps.")
        steps = '\n'.join(steps)
        ex = f"### Example {index} ({config['task_id']}): {instruction}\n{steps}"
    else:
        ex = None
    return ex, task


def get_example_query_cleaned(index: int, result_dir: str, config_dir: str, benchmark: str = "webarena") -> str:
    """Get the string for a past example experience, without invalid actions."""
    # get config
    cid = get_task_id(result_dir)
    config_path = os.path.join(config_dir, benchmark, f"{cid}.json")
    config = json.load(open(config_path))

    # get instruction
    subtask_inst_path = os.path.join(result_dir, "instruction.txt")
    if os.path.exists(subtask_inst_path):  # sub task
        instruction = open(subtask_inst_path, 'r').read()
        task = instruction
    else:  # full task
        instruction = config["intent"]
        task = config["intent_template"]

    steps = json.load(open(os.path.join(result_dir, "cleaned_steps.json")))
    if len(steps) > 0:
        print(f"Collected #{len(steps)} valid steps.")
        steps = '\n'.join(steps)
        ex = f"### Example {index} ({config['task_id']}): {instruction}\n{steps}"
    else:
        ex = None

    return ex, task

def get_test_query(result_dir_list: str, config_dir: str, benchmark: str = "webarena") -> str:
    """Transform past examples into a test query."""
    task = None
    examples = []
    for rdir in result_dir_list:
        ex, task = get_example_query_cleaned(len(examples)+1, rdir, config_dir, benchmark)
        if ex is None: continue
        examples.append(ex)
    
    if len(examples) < 1:
        return None
    query = f"## Task: {task}\n" + '\n\n'.join(examples)
    return query


def induce_actions(force_one_rewritten_trajectory: bool = False) -> list[str] | None:
    result_dir_list = get_result_dirs(args.results_dir, args.result_id_list, args.template_id, args.config_dir, args.benchmark)
    test_query = get_test_query(result_dir_list, args.config_dir, args.benchmark)
    if test_query is None: return []
    with open(args.test_query_path, 'w') as fw: fw.write(test_query)

    messages = [{"role": "system", "content": open(args.sys_msg_path).read()}]
    messages += [{"role": "user", "content": open(args.instruction_path).read()}]
    messages += [{"role": "user", "content": open(args.few_shot_path).read()}]
    if force_one_rewritten_trajectory:
        messages += [{"role": "user", "content": "You must only have one code block under the 'Rewritten Trajectories' heading. Even if you can rewritte the trajectory in multiple ways or variants, you must choose and only write one of them."}]
    messages += [{"role": "user", "content": test_query + '\n\n## Reusable Functions'}]

    all_responses = []
    # Both vLLM and OpenRouter support n parameter
    response = llm_completion(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        n=args.num_responses,
        usage_name=f"induce_actions for task {' '.join(args.result_id_list)}"
    )
    contents = get_contents(response, remove_thinking="qwen" in args.model)
    for i, curr_content in enumerate(contents):
        curr_path = os.path.join(args.output_dir, f"{i}.md")
        with open(curr_path, 'w') as fw:
            fw.write(test_query + '\n' + curr_content)
        all_responses.append(curr_content)
    return all_responses



# %% Process Actions
from induce.utils import count_function_calls, get_function_names

def write_actions(response: str) -> tuple[str, list[str]]:
    """Extract actions from response and write actions to agent action loading file."""
    existing_action_names = get_function_names(open(args.write_action_path, 'r').read())
    # extract induced actions from the response
    if "reusable functions" in response.lower():
        index = response.lower().rindex("reusable functions")
        response = response[index:].lstrip()
    elif "reusable function" in response.lower():
        index = response.lower().rindex("reusable function")
        response = response[index:].lstrip()
    actions = extract_code_pieces(response, start="```python", end="```", do_split=False)
    actions = [a for a in actions if "def " in a and count_function_calls(a, 1)]
    if len(actions) == 0:  # A fallback for the case that the actions are not wrapped in ```python, since we check the string after "reusable functions", it is safe to take the string that is wrapped in ```, sometimes the LLM response is not well formatted.
        response = response.replace('```python', '```')
        actions = extract_code_pieces(response, start="```", end="```", do_split=False)
        actions = [a for a in actions if "def " in a and count_function_calls(a, 1)]
    new_actions, action_names = [], []
    for a in actions:
        if ("def " in a) and count_function_calls(a, 1):
            a_names = get_function_names(a, existing_action_names)
            if len(a_names) > 0:
                action_names.extend(a_names)
                new_actions.append(a)

    print(
        f"Induced #{len(new_actions)}|{len(action_names)} Actions, ",
        [next(line for line in a.split("\n") if "def " in line) for a in new_actions],
        action_names
    )
    if len(new_actions) == 0: return None, None

    tmp_path = args.write_action_path + ".tmp"
    process = subprocess.Popen(["cp", args.write_action_path, tmp_path])
    process.wait()

    with open(args.write_action_path, 'a+') as fw:
        fw.write('\n\n'+ '\n\n'.join(new_actions))
    return tmp_path, action_names



# %% Run Tests
from induce.utils import parse_tests

def write_tests(response: str, result_id_list: list[str], action_names: list[str] = [], benchmark: str = "webarena") -> bool:
    """Extract tests, write tests to file, and run tests.
    Args:
        response: model generated response including induced actions, tests, and texts.
        result_id_list: list of task result IDs.
        action_names: list of names of the induced actions to test on.
    Returns:
        bool: If all tests passed the check.
    """
    tests = parse_tests(response, action_names)
    assert len(tests) == len(result_id_list), f"Got #{len(tests)} tests but for #{len(result_id_list)} results."
    
    # write tests and run them
    for i, (t, r) in enumerate(zip(tests, result_id_list)):
        # write test trajectory
        if not os.path.exists(args.write_tests_dir):
            os.makedirs(args.write_tests_dir)
        test_path = os.path.join(args.write_tests_dir, f"test_{i}.txt")
        test_str = '\n'.join([f"```{tl.strip()}```" for tl in t.split('\n') if tl.strip()])
        with open(test_path, 'w') as fw:  # overwrite existing content
            fw.write(test_str)
        
        task_name = f"{benchmark}.{r.split('_')[0]}"
        print(f"Running test {i} for task {task_name}...")
        # run test
        done = run_demo_task(
            task_name=task_name,
            websites=args.website,
            rename_to=f"{task_name}_test",
            action_path=test_path,
            model_name=args.model,
            results_dir=args.results_dir,
            headless=True,
            usage_prefix=f"(induce_actions)"
        )
        if not done:
            return True  # revert on timeout

    # check test results
    scores = []
    for r in result_id_list:
        if args.eval_with_gold:
            eval_path = os.path.join(args.results_dir, f"{benchmark}.{r}_test", "summary_info.json")
            if os.path.exists(eval_path):
                scores.append(json.load(open(eval_path))["cum_reward"] == 1.0)
            else:
                scores.append(False)
        else:
            is_correct = evaluate_trajectory(
                f"{benchmark}.{r}_test",
                model_name=args.eval_model_name,
                results_dir=args.results_dir,
                config_dir=args.config_dir,
            )
            scores.append(is_correct)
            if not is_correct:
                print(f"Test of the induced function for {task_name} is judged as incorrect.")
            else:
                print(f"Test of the induced function for {task_name} is judged as correct.")

        # check step valid and use actions
        print("\nChecking validity of steps after the agent ran the test for verification...")
        output = calc_valid_steps(f"{benchmark}.{r}_test", args.results_dir, model_name=args.model, action_names=action_names)
        output = output.split('\n')[-1].strip()
        print("Validity Check: ", output)
        if output == 'False': scores[-1] = False

        if scores[-1] == False: break
    
    print("Scores: ", scores)
    if all([s == True for s in scores]):
        print("All Tests Passed!")
        return False
    else:
        return True


# %% Overall pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openrouter/google/gemini-3-flash-preview")
    parser.add_argument("--eval_model_name", type=str, default="openrouter/openai/gpt-4o")
    parser.add_argument("--num_responses", type=int, default=1, help="Number of responses to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")

    parser.add_argument("--sys_msg_path", type=str, default="induce/prompt/system_message.txt")
    parser.add_argument("--instruction_path", type=str, default="induce/prompt/instruction.txt")
    parser.add_argument("--few_shot_path", type=str, default="induce/prompt/shopping.md")
    parser.add_argument("--test_query_path", type=str, default="induce/prompt/test_query.txt")

    parser.add_argument("--template_id", type=str, default=None)
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "admin", "reddit", "gitlab", "map"])
    parser.add_argument("--benchmark", type=str, default="webarena")
    parser.add_argument("--config_dir", type=str, default="config_files/default", 
                        help="Task config root (config_files/<run_id>); default is rewritten to match CONSTBUDG_RUN_SUFFIX.")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--result_id_list", type=str, nargs="+", default=None, help="E.g., '110 111'.")

    parser.add_argument("--write_action_path", type=str, default=None)
    parser.add_argument("--write_tests_dir", type=str, default="induce/default/debug_actions")
    parser.add_argument("--eval_with_gold", action="store_true", help="If perform evaluation with ground-truth.")
    args = parser.parse_args()

    # decide path to write actions
    if args.write_action_path is None:
        args.write_action_path = os.path.join("actions", f"{args.website}.py")
    apply_induce_actions_run_suffix_paths(args)

    # decide path for entire model output
    args = get_output_dir(args)
    os.makedirs(args.output_dir, exist_ok=True)
    responses = induce_actions()

    # It happens that the LLM output multiple rewritten trajectories. In that case, an assert fails in write_tests.
    # So we induce actions again with force_one_rewritten_trajectory=True to prevent that.
    # This flag is not True by default to avoid diverging from the original pipeline, and is used only in case of failure.
    if len(responses) > 0:
        tests_count = len(parse_tests(responses[0], action_names=None))
        if tests_count != len(args.result_id_list):
            print(f"Warning: Got {tests_count} tests under 'rewritten trajectories' heading but for {len(args.result_id_list)} results.")
            print(f"Inducing actions with force_one_rewritten_trajectory=True...")
            responses = induce_actions(force_one_rewritten_trajectory=True)
    
    # write actions and run tests
    print(f"Collected {len(responses)} Responses..")
    for i, resp in enumerate(responses):
        print(f"\n\n** Start Evaluating Response {i} **\n", resp, "\n")
        tmp_path, action_names = write_actions(resp)
        if tmp_path is None: continue

        if_revert = write_tests(resp, args.result_id_list, action_names, args.benchmark)
        print("If Revert: ", if_revert)
        if if_revert:
            process = subprocess.Popen(["mv", tmp_path, args.write_action_path])
            process.wait()
            for i, r in enumerate(args.result_id_list):
                print("Command: ", ["rm", "-rf", f"{args.results_dir}/{args.benchmark}.{r}_test"])
                process = subprocess.Popen(["rm", "-rf", f"{args.results_dir}/{args.benchmark}.{r}_test"])
                process.wait()
        else:
            process = subprocess.Popen(["rm", tmp_path])
            process.wait()
            print(f"Test passed and the function is written to {args.write_action_path}!")
            break
            
        print(f"** Finish Evaluating Response {i} **\n\n")
        