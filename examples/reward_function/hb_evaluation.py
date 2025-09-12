import json
import os
import re
from typing import Dict, List, Set
from examples.reward_function.detailed_multi_task_evaluation import evaluate_predictions, _build_index_to_label
import datetime

# Function to extract boxed content from model predictions
def extract_boxed(text: str) -> str:
    """Extract content from \\boxed{...} format, or return original text."""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return text.strip()

def compute_metrics_by_data_source(
    predictions: List[str],
    ground_truths: List[str],
    datasets: List[str],
    save_path: str = None,
    global_steps: int = None,
) -> Dict[str, float]:
    """
    Compute metrics at the dataset level and a global mean across datasets.
    This is now a wrapper around the newer evaluate_predictions function.
    """
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    input_data = {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "datasets": datasets,
    }
    # name is time in yyyy-mm-dd_hh-mm-ss format
    with open(
            os.path.join(output_dir, f"input_data_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"), "w"
    ) as f:
        json.dump(input_data, f, indent=4)
    # Get the absolute path to meta.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    label_map_path = os.path.join(script_dir, "meta.json")

    ignore_datasets = {"mimeqa", "siq2", "intentqa"}
    # remove predictions, ground_truths, datasets for ignored datasets
    filtered_predictions = []
    filtered_ground_truths = []
    filtered_datasets = []
    for pred, gt, ds in zip(predictions, ground_truths, datasets):
        if ds not in ignore_datasets:
            filtered_predictions.append(pred)
            filtered_ground_truths.append(gt)
            filtered_datasets.append(ds)
    predictions = filtered_predictions
    ground_truths = filtered_ground_truths
    datasets = filtered_datasets
    
    # Load meta config to build label mappings
    with open(label_map_path, "r") as f:
        cfg = json.load(f)
    meta_config = cfg["meta"]
    
    # Build index -> label and label -> index mappings
    index_to_label = _build_index_to_label(meta_config)
    label_to_index = {v.lower(): k for k, v in index_to_label.items()}
    
    # Convert string predictions to integers
    predictions_int = []
    for pred in predictions:
        # Extract boxed content if present
        pred_clean = extract_boxed(pred)
        
        # Try to convert to index
        pred_lower = pred_clean.lower()
        if pred_lower in label_to_index:
            predictions_int.append(label_to_index[pred_lower])
        else:
            # If not found in mapping, try to handle as a raw string
            # Default to -2 if completely unrecognizable
            predictions_int.append(-2)
    
    # Convert string ground truths to integers
    ground_truths_int = []
    for gt in ground_truths:
        gt_lower = gt.lower()
        if gt_lower in label_to_index:
            ground_truths_int.append(label_to_index[gt_lower])
        else:
            # Try without spaces or special characters
            gt_clean = gt_lower.replace(" ", "").replace("_", "")
            if gt_clean in label_to_index:
                ground_truths_int.append(label_to_index[gt_clean])
            else:
                # Default to -1 if unrecognizable
                print("Warning: Ground truth label not found in mapping:", gt)
                ground_truths_int.append(-1)
    
    # Call the new evaluate_predictions function
    results = evaluate_predictions(
        predictions=predictions_int,
        ground_truths=ground_truths_int,
        datasets=datasets,
        save_path=save_path,
        global_steps=global_steps,
        label_map_path=label_map_path
    )
    
    # Format the return value to match the original interface
    result = {}
    
    # Add per-dataset metrics
    if "per_dataset_metrics" in results:
        result.update(results["per_dataset_metrics"])
    
    # Add averaged metrics from aggregate metrics
    if "aggregate_metrics" in results:
        for k, v in results["aggregate_metrics"].items():
            result[f"val/averaged_{k}"] = v
    
    return result

if __name__ == "__main__":
    # predictions   = [
    #     # DatasetA (6 samples)
    #     "<think>Well, looking at the image, the person's eyes are looking down and their mouth is in a sort of frown. There's no big smile or anything that would suggest happiness. It doesn't look like they're angry or fearful either. So, I'd say the primary facial expression is sad.</think> \\boxed{sad}", "A", "B", "B", "A", "B",
    #     # DatasetB (5 samples)
    #     "A", "C", "C", "B", "B",
    # ]
    # ground_truths = [
    #     # DatasetA GT
    #     "sad", "B", "B", "B", "A", "A",
    #     # DatasetB GT
    #     "A", "C", "B", "B", "C",
    # ]
    # datasets = [
    #     "DatasetA","DatasetA","DatasetA","DatasetA","DatasetA","DatasetA",
    #     "DatasetB","DatasetB","DatasetB","DatasetB","DatasetB",
    # ]

    input_filename = "examples/reward_function/omni_all_modalities.json"
    with open(input_filename, "r") as f:
        input_data = json.load(f)
    predictions = input_data["predictions"]
    ground_truths = input_data["ground_truths"]
    datasets = input_data["datasets"]

    metrics = compute_metrics_by_data_source(predictions, ground_truths, datasets)
    # drop sub metrics (with 2 slashes)
    metrics = {k: v for k, v in metrics.items() if k.count('/') < 2}
    print(metrics)
    # print("=== Synthetic two-dataset sanity check ===")
    # print(json.dumps(metrics, indent=4))