import os
from browsergym.experiments.benchmark.metadata.utils import task_metadata


def parse_task_ids(task_id_str: str) -> list[str]:
    chunks = [c.strip() for c in task_id_str.split(",")]
    task_id_list = []
    for c in chunks:
        if "." in c:
            continue
        nums = [int(n.strip()) for n in c.split("-")]
        if len(nums) == 1:
            task_id_list.append(str(nums[0]))
        elif len(nums) == 2:
            s, e = nums
            task_id_list.extend([str(i) for i in range(s, e+1)])
    return task_id_list


def get_task_names(
    benchmark: str, website: str, task_ids: str,
) -> list[str]:
    if website == "admin":
        website = "shopping_admin"
    
    urls = {
        "gitlab": os.environ.get("WA_GITLAB"),
        "shopping": os.environ.get("WA_SHOPPING"),
        "shopping_admin": os.environ.get("WA_SHOPPING_ADMIN"),
        "reddit": os.environ.get("WA_REDDIT"),
        "wikipedia": os.environ.get("WA_WIKIPEDIA"),
        "map": os.environ.get("WA_MAP"),
        "homepage": os.environ.get("WA_HOMEPAGE"),
    }
    available_sites = [k for k, v in urls.items() if v and str(v).lower() != "todo"]
    print(f"Available websites: {available_sites}")

    if website == "multi":  # multi-website case
        requested_websites = ["admin", "shopping", "reddit", "gitlab", "map"]
        print(f"Selecting tasks that involve multiple websites.")
    else:
        requested_websites = [website]
        print(f"Selecting tasks that only involve {website}.")
    for w in requested_websites:
        assert w in available_sites, f"Website {w} not found in available websites, have you exported the environment variables?"

    metadata = task_metadata(benchmark)
    initial_ids = parse_task_ids(task_ids)
    valid_task_names = []
    
    for tid in initial_ids:
        row = metadata[metadata["task_id"] == int(tid)]
        if row.empty: 
            print(f"Task {tid} not found in metadata.")
            continue
        task_sites = row.iloc[0]["sites"].split()
        if (website == "multi" and len(task_sites) > 1) or (website in task_sites and len(task_sites) == 1):
            valid_task_names.append(row.iloc[0]["task_name"])

    if not valid_task_names:
        print(f"Warning: No valid tasks found for available websites in the requested range.")
    else:
        print(f"Tasks to evaluate on: {valid_task_names}")
    return valid_task_names
