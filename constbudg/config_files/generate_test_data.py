"""Replace the website placeholders with website domains from environment variables
Generate the test data
"""

import os
import json
import argparse

PLACEHOLDER_ENV_MAP = {
    "__SHOPPING__": "WA_SHOPPING",
    "__SHOPPING_ADMIN__": "WA_SHOPPING_ADMIN",
    "__REDDIT__": "WA_REDDIT",
    "__GITLAB__": "WA_GITLAB",
    "__MAP__": "WA_MAP",
    "__WIKIPEDIA__": "WA_WIKIPEDIA",
    "__HOMEPAGE__": "WA_HOMEPAGE",
}


def load_env_file(env_path: str) -> dict:
    """Load environment variables from a .env file.
    
    Args:
        env_path: Path to the .env file
        
    Returns:
        Dictionary of key-value pairs from the .env file
    """
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=value format
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")  # Remove quotes if present
                    env_vars[key] = value
    return env_vars


def get_env_value(env_var: str, env_file_vars: dict) -> str:
    """Get environment variable value, checking .env file if not in os.environ.
    
    Args:
        env_var: Name of the environment variable
        env_file_vars: Dictionary of variables loaded from .env file
        
    Returns:
        Value of the environment variable, or None if not found
    """
    # First check os.environ
    url = os.environ.get(env_var)
    if url:
        return url
    
    # If not found, check .env file
    return env_file_vars.get(env_var)


def main(input_file: str, output_dir: str) -> None:
    """Generate config files from raw JSON template.
    
    Args:
        input_file: Path to the raw JSON template file (e.g., test.raw.json)
        output_dir: Directory where individual task config files will be written
    """
    # Load .env file from repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(repo_root, ".env")
    env_file_vars = load_env_file(env_path)
    
    with open(input_file, "r") as f:
        raw = f.read()
    
    # Replace placeholders with URLs from environment variables
    for placeholder, env_var in PLACEHOLDER_ENV_MAP.items():
        url = get_env_value(env_var, env_file_vars)
        if url and str(url).lower() != "todo":
            raw = raw.replace(placeholder, url)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        f.write(raw)

    # Split to multiple files
    data = json.loads(raw)
    for idx, item in enumerate(data):
        output_path = os.path.join(output_dir, f"{idx}.json")
        with open(output_path, "w") as f:
            json.dump(item, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate config files from raw JSON template by replacing website placeholders"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="config_files/test.raw.json",
        help="Path to the raw JSON template file (default: config_files/test.raw.json, for webarena)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="config_files/webarena",
        help="Directory where individual task config files will be written (default: config_files/webarena)"
    )
    
    args = parser.parse_args()
    main(args.input, args.output_dir)
