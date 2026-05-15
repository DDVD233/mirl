"""
Clones a HuggingFace model repo and patches model_type/architectures in
config.json so that vLLM can load it with its native Qwen2.5-Omni
implementation instead of the unsupported Thinker fallback.

Usage:
    python patch_hf_config.py \\
        --source keentomato/harpo_hier_step400 \\
        --target keentomato/harpo_hier_step400_omni
"""

import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="keentomato/harpo_hier_step400",
                        help="Source HF repo to clone from")
    parser.add_argument("--target", required=True,
                        help="Target HF repo to upload the patched model to")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Make the target repo private (default: True)")
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()

    print(f"Creating target repo {args.target} ...")
    api.create_repo(repo_id=args.target, repo_type="model", private=args.private, exist_ok=True)

    print(f"Downloading all files from {args.source} ...")
    local_dir = snapshot_download(repo_id=args.source, repo_type="model")
    print(f"  Downloaded to {local_dir}")

    config_path = f"{local_dir}/config.json"
    with open(config_path) as f:
        config = json.load(f)

    old_type = config.get("model_type")
    old_arch = config.get("architectures")
    print(f"  model_type:    {old_type} -> qwen2_5_omni")
    print(f"  architectures: {old_arch} -> ['Qwen2_5OmniForConditionalGeneration']")

    config["model_type"] = "qwen2_5_omni"
    config["architectures"] = ["Qwen2_5OmniForConditionalGeneration"]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Uploading all files to {args.target} ...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=args.target,
        repo_type="model",
        commit_message=f"clone of {args.source} with model_type patched to qwen2_5_omni",
    )
    print("Done.")

if __name__ == "__main__":
    main()
