import os
import json
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_client import llm_completion, get_contents
from induce.utils import get_output_dir, get_task_id, get_result_dirs
from utils.run_context import apply_induce_memory_run_suffix_paths

# %% Induce Memory

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
    

def get_test_query(result_dir_list: str, config_dir: str, benchmark: str = "webarena") -> str | None:
    """Transform each result log into an input experience and form a query."""
    task = None
    examples = []
    for rdir in result_dir_list:  # e.g., ['results/webarena.110_2_0', 'results/webarena.111_2_1']
        ex, task = get_example_query_cleaned(len(examples)+1, rdir, config_dir, benchmark)
        if ex is None: continue
        examples.append(ex)
    
    if len(examples) < 1:
        return None
    query = f"## Task: {task}\n" + '\n\n'.join(examples)
    return query


def induce_workflows() -> list[str]:
    result_dir_list = get_result_dirs(args.results_dir, args.result_id_list, args.template_id, args.config_dir)
    test_query = get_test_query(result_dir_list, args.config_dir, args.benchmark)
    if test_query is None: return []
    with open(args.test_query_path, 'w') as fw:
        fw.write(test_query)

    messages = [{"role": "system", "content": open(args.sys_msg_path).read()}]
    messages += [{"role": "user", "content": open(args.instruction_path).read()}]
    messages += [{"role": "user", "content": open(args.few_shot_path).read()}]
    messages += [{"role": "user", "content": "## Existing Workflows\n" + open(args.write_workflow_path).read()}]
    messages += [{"role": "user", "content": test_query + '\n\n## Reusable Workflows'}]

    all_responses = []
    # Both vLLM and OpenRouter support n parameter
    response = llm_completion(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        n=args.num_responses,
        usage_name=f"induce_memory for task {' '.join(args.result_id_list)}"
    )
    contents = get_contents(response, remove_thinking="qwen" in args.model)
    for i, curr_content in enumerate(contents):
        curr_path = os.path.join(args.output_dir, f"{i}.md")
        with open(curr_path, 'w') as fw:
            fw.write(test_query + '\n\n\n' + curr_content)
        all_responses.append(curr_content)
    return all_responses

# %% Write Workflows
from induce.utils import extract_code_pieces

def get_workflow_name(workflow: str) -> str:
    """Get the name of the workflow."""
    name = workflow.split('\n')[0].lstrip("Task: ").strip()
    return name


def update_workflows(workflow: str, existing_workflows: list[str]) -> tuple[bool, list[str]]:
    """Update the existing workflows given the potentially topically similar new item.
    - If the new workflow does not overlap with any existing workflows, add it => True, []
    - If the new workflow topically overlap:
      - If the new workflow is better, replace the existing workflow => True, [existing_workflow_name]
        - If the existing workflow is better, keep the existing workflow => False, []
    """
    name = get_workflow_name(workflow)
    for ew in existing_workflows:
        ew_name = get_workflow_name(ew)
        messages = [
            {"role": "system", "content": "You are an expert in navigating the web, your task is to check if the two workflows refer to the same task."},
            {"role": "user", "content": "Does the following two workflows refer to the same task? Only return 'yes' or 'no', do not provide any additional information."},
            {"role": "user", "content": f"Workflow 1: {name}\nWorkflow 2: {ew_name}"}
        ]
        response = llm_completion(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            usage_name=f"update_workflows for task {' '.join(args.result_id_list)}"
        )
        response = get_contents(response, remove_thinking="qwen" in args.model)[0]
        
        if 'yes' in response: yes_index = response.index('yes')
        else: yes_index = 0
        if 'no' in response: no_index = response.index('no')
        else: no_index = len(response)
        if yes_index < no_index:
            print(f"Checking Overlap between [{name}] & [{ew_name}] => YES")
            # if the existing workflow is better, still count as overlap
            better_workflow = get_better_workflow(workflow, ew)
            action = "KEEP" if better_workflow == ew else "REPLACE"
            print(f"Better Workflow: {better_workflow} \n=> {action}")
            if better_workflow == ew: # existing workflow is better
                return False, []
            else:  # new workflow is better
                return True, [ew_name]
    print(f"Checking Overlap between [{name}] & [{len(existing_workflows)} Existing Workflows] => NO")
    return True, []


def get_better_workflow(workflow1: str, workflow2: str) -> str:
    """Select the better workflow between two topically-overlapping workflows."""
    messages = [
        {"role": "system", "content": "You are an expert in navigating the web, your task is to select the better navigation guidance workflow between the two workflows provided."},
        {"role": "user", "content": "Which workflow is more helpful in guiding web navigation? Only return 'Workflow 1' or 'Workflow 2', do not provide any additional information."},
        {"role": "user", "content": f"Workflow 1:\n{workflow1}\nWorkflow 2:\n{workflow2}"}
    ]
    response = llm_completion(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        usage_name=f"get_better_workflow for task {' '.join(args.result_id_list)}"
    )
    response = get_contents(response, remove_thinking="qwen" in args.model)[0]
    if "workflow 1" in response.lower(): return workflow1
    elif "workflow 2" in response.lower(): return workflow2
    else: return None

def write_workflows(response: str) -> None:
    # get newly induced workflows
    workflows = extract_code_pieces(response, start='"""', end='"""', do_split=False)
    if len(workflows) == 0 and "Task:" in response:
        print('Warning: No workflows found in the response after splitting by """. Trying to extract from the response text...')
        workflows = ["Task:" + w.strip() for w in response.split("Task:") if len(w.strip()) > 0]
    workflows = [w for w in workflows if ("Task" in w) and ("Action Trajectory" in w)]

    # load existing workflows
    existing_workflows = open(args.write_workflow_path, 'r').read().split("Task:")
    existing_workflows = ["Task:"+w for w in existing_workflows if len(w) > 0]
    existing_workflows = [w.strip() for w in existing_workflows]
    existing_workflows = [w for w in existing_workflows if len(w) > 0]
    existing_workflow_names = [get_workflow_name(w) for w in existing_workflows]

    # update workflows
    new_workflows = []
    for w in workflows:
        add_new, names_to_remove = update_workflows(w, existing_workflows)
        existing_workflows = [
            ew for n,ew in zip(existing_workflow_names, existing_workflows)
            if n not in names_to_remove
        ]
        if add_new: new_workflows.append(w)

    # rewrite the entire workflow memory
    with open(args.write_workflow_path, 'w') as fw:
        fw.write('\n\n'.join(existing_workflows + new_workflows))


# %% Overall pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openrouter/google/gemini-3-flash-preview")
    parser.add_argument("--num_responses", type=int, default=1, help="Number of responses to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")

    parser.add_argument("--sys_msg_path", type=str, default="induce/prompt/system_message_memory.txt")
    parser.add_argument("--instruction_path", type=str, default="induce/prompt/instruction_memory.txt")
    parser.add_argument("--few_shot_path", type=str, default="induce/prompt/shopping_memory.md")
    parser.add_argument("--test_query_path", type=str, default="induce/prompt/test_query.txt")

    parser.add_argument("--template_id", type=str, default=None)
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "admin", "reddit", "gitlab", "map"])
    parser.add_argument("--benchmark", type=str, default="webarena")
    parser.add_argument("--config_dir", type=str, default="config_files/default", 
                        help="Task config root (config_files/<run_id>); default is rewritten to match CONSTBUDG_RUN_SUFFIX.")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--result_id_list", type=str, nargs="+", default=None, help="E.g., '110_2_0 111_1'.")

    parser.add_argument("--write_workflow_path", type=str, default=None)
    parser.add_argument("--write_tests_dir", type=str, default="induce/debug_memory")
    parser.add_argument("--eval_with_gold", action="store_true")
    args = parser.parse_args()

    if args.write_workflow_path is None:
        args.write_workflow_path = os.path.join("workflows", f"{args.website}.txt")
    apply_induce_memory_run_suffix_paths(args)

    # decide path for entire model output
    args = get_output_dir(args, key="workflow")
    os.makedirs(args.output_dir, exist_ok=True)
    responses = induce_workflows()
        
    # write actions and run tests
    for i, resp in enumerate(responses):
        print(f"\n\n** Start Evaluating Response {i} **\n", resp, "\n")
        write_workflows(resp)

        print(f"** Finish Evaluating Response {i} **\n\n")
