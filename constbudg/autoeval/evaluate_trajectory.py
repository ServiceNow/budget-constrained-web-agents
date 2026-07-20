import os
import json
import argparse
import traceback
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.run_context import run_segment_from_config_dir

from autoeval.evaluator import Evaluator
from autoeval.clients import CLIENT_DICT


def load_blocks(path: str) -> list[str]:
    blocks, curr_block = [], []
    for line in open(path, 'r'):
        if "INFO" in line:
            if curr_block:
                blocks.append(curr_block)
                curr_block = []
        else:
            curr_block.append(line.strip())
    if curr_block:
        blocks.append(curr_block)
    blocks = ['\n'.join(b) for b in blocks]
    blocks = [b for b in blocks if "```" in b]
    return blocks
        

def remove_invalid_steps(actions: list[str]) -> list[str]:
    """Remove invalid steps from the action sequence."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            if type(eval(arg)) == str:
                valid_actions.append(a)
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            if type(eval(arg)) == str:
                valid_actions.append(a)
        else:
            valid_actions.append(a)
    return valid_actions


def extract_code_pieces(text: str) -> list[str]:
    """Extract code pieces from a text string.
    Args:
        text: str, model prediction text.
    Rets:
        code_pieces: list[str], code pieces in the text.
    """
    code_pieces = []
    while "```" in text:
        st_idx = text.index("```") + len("```")
        if "```" in text[st_idx:]:
            end_idx = text.index("```", st_idx + 1)
        else: 
            end_idx = len(text)
        code_pieces.append(text[st_idx:end_idx].strip())
        text = text[end_idx+3:].strip()
    return code_pieces


def extract_think_and_action(path: str) -> tuple[list[str], list[str]]:
    blocks = load_blocks(path)
    think_list, action_list = [], []
    for b in blocks:
        sidx = b.index('```')
        think_list.append(b[:sidx].strip())
        actions = extract_code_pieces(b[sidx:])
        action_list.extend(actions)
    return think_list, action_list


def extract_response(action: str) -> str:
    s, e = action.index("(")+1, action.index(")")
    return action[s: e]


def process_sample(
    idx: str, traj_info: dict, log_save_path,
    model: str, eval_version: str,
) -> list[dict]:
    actual_model_name = model.split("/")[-1]
    clients = {model: CLIENT_DICT[actual_model_name](model_name=model)}
    evaluator = Evaluator(clients, log_save_path=log_save_path + "/trajs")
    try:
        out, _ = evaluator(traj_info, model, eval_version)
        eval_result = None
        if out["status"].lower() == "success": eval_result = True
        else: eval_result = False
        return [{
                "idx": idx,
                "gt": traj_info["eval"],
                "rm": eval_result,
                "thoughts": out["thoughts"], 
                "uid": traj_info["traj_name"],
        }]
    except Exception as e:
        print(f"Error on {idx}, {e}")
        print(traceback.format_exc())
        return [{
            "idx": idx,
            "gt": traj_info["eval"],
            "rm": None,
            "thoughts": None, 
            "uid": traj_info["traj_name"],
        }]


def main():
    # load task config
    benchmark, task_id = args.result_dir.split('/')[-1].split('.')
    if '_' in task_id:
        task_id = task_id.split('_')[0]
    config_path = os.path.join(args.config_dir, benchmark, f"{task_id}.json")
    config = json.load(open(config_path))

    # load trajectory log
    log_path = os.path.join(args.result_dir, "experiment.log")
    think_list, action_list = extract_think_and_action(log_path)
    if len(action_list) == 0:
        raise ValueError(f"No actions found in {log_path}, please check the trajectory.")
    if "send_msg_to_user" in action_list[-1]:
        response = extract_response(action_list[-1])
    else:
        response = ""
    
    # load summary info
    summary_path = os.path.join(args.result_dir, "summary_info.json")
    summary = json.load(open(summary_path, 'r'))

    # collect traj info
    image_paths = [
        os.path.join(args.result_dir, f) for f in os.listdir(args.result_dir) 
        if f.startswith("screenshot_step_") and f.endswith(".png")
    ]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split("_")[-1].split(".")[0]))

    subtask_dir = args.result_dir.rstrip("_test")
    inst_path = os.path.join(subtask_dir, "instruction.txt")
    if os.path.exists(inst_path):
        intent = open(inst_path, 'r').read().strip()
    else:
        intent = config["intent"]
    print("\n")
    print("="*50)
    print("Intent:", intent)
    print("="*50)
    traj_info = {
        "intent": intent,
        "response": response,
        "captions": think_list,
        "actions": action_list,
        "traj_name": config["task_id"],
        "image_paths": image_paths,
        "images": image_paths,
        "eval": summary["cum_reward"]
    }

    task_leaf = args.result_dir.split("/")[-1]
    run_seg = run_segment_from_config_dir(args.config_dir)
    log_save_path = os.path.join("autoeval", "log", run_seg, task_leaf)
    print("Log Save Path:", log_save_path)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
        os.makedirs(log_save_path + "/trajs")
    eval_info = process_sample(
        idx=config["task_id"], traj_info=traj_info,
        log_save_path=log_save_path, 
        model=args.model, eval_version=args.prompt,
    )
    actual_model_name = args.model.split("/")[-1]
    output_eval_path = os.path.join(args.result_dir, f"{actual_model_name}_autoeval.json")
    json.dump(eval_info, open(output_eval_path, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to the result directory, e.g., 'webarena.0'.")
    parser.add_argument("--config_dir", type=str, default="config_files/default",
                        help="Task config root (e.g. config_files/<run_id>); logs under autoeval/log/<segment>/...")
    # autoeval
    parser.add_argument("--model", type=str, default="openrouter/openai/gpt-4o")
    parser.add_argument("--prompt", type=str, default="vision",
                        choices=["text", "vision"])

    args = parser.parse_args()

    if args.model == "gpt-4o" and args.prompt != "vision":
        print(f"Waring: use vision prompt by default for {args.model}.")
        args.prompt = "vision"

    main()
