import shutil
from pathlib import Path
import wandb

# ---- config ----
PROJECT = "harpo_task_only_advantage_distribution_upload"
ARTIFACT_NAME = "harpo_task_only_advantage_distribution"
ARTIFACT_TYPE = "dataset"

FOLDER_TO_ZIP = Path("/scratch/keane/human_behaviour/tarpo_iter_22_no_resp_no_hierarchy_mixtures_densities_ema_original/advantages")  # <-- set this
ZIP_PATH = Path("/scratch/keane/human_behaviour/tarpo_iter_22_no_resp_no_hierarchy_mixtures_densities_ema_original/harpo_task_only_advantages.zip")        # <-- output zip

# ---- zip folder ----
if not FOLDER_TO_ZIP.exists() or not FOLDER_TO_ZIP.is_dir():
    raise FileNotFoundError(f"Folder does not exist or is not a directory: {FOLDER_TO_ZIP}")

# shutil.make_archive wants the base name WITHOUT ".zip"
ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
base_name = str(ZIP_PATH.with_suffix(""))  # /path/to/hier_advantages
archive_path = shutil.make_archive(base_name=base_name, format="zip", root_dir=str(FOLDER_TO_ZIP))

# make_archive returns the final filename
archive_path = Path(archive_path)
print(f"Created zip: {archive_path} ({archive_path.stat().st_size / 1e6:.2f} MB)")

# ---- upload to W&B as artifact ----
run = wandb.init(project=PROJECT)

artifact = wandb.Artifact(name=ARTIFACT_NAME, type=ARTIFACT_TYPE)
artifact.add_file(str(archive_path))

run.log_artifact(artifact)
run.finish()