import json
from typing import Dict, List, Any
from collections import defaultdict


def load_step_file(step_file_path: str) -> Dict[str, List]:
    """
    Load the step JSON file containing predictions, ground_truths, and datasets.

    Args:
        step_file_path: Path to the step JSON file

    Returns:
        Dictionary with 'predictions', 'ground_truths', and 'datasets' keys
    """
    with open(step_file_path, 'r') as f:
        data = json.load(f)

    return data


def load_jsonl_file(jsonl_file_path: str) -> List[Dict[str, Any]]:
    """
    Load the JSONL file where each line is a JSON object.

    Args:
        jsonl_file_path: Path to the JSONL file

    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    return data


def group_step_data_by_dataset(step_data: Dict[str, List]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group step data by dataset, preserving the order within each dataset chunk.

    Args:
        step_data: Dictionary with 'predictions', 'ground_truths', and 'datasets' keys

    Returns:
        Dictionary mapping dataset name to list of samples in order
    """
    predictions = step_data['predictions']
    ground_truths = step_data['ground_truths']
    datasets = step_data['datasets']

    # Group by dataset while preserving order
    dataset_groups = defaultdict(list)

    for i, dataset in enumerate(datasets):
        dataset_groups[dataset].append({
            'prediction': predictions[i],
            'ground_truth': ground_truths[i],
            'original_index': i
        })

    return dataset_groups


def merge_files(step_file_path: str, jsonl_file_path: str) -> List[Dict[str, Any]]:
    """
    Merge step file and JSONL file, aligning samples by dataset.

    The step file has samples grouped in chunks by dataset (where datasets can appear
    multiple times in different chunks). The JSONL file has samples grouped by dataset
    in a specific order. This function merges them so that:
    1. Samples from the same dataset in step file (across chunks) are combined in order
    2. The merged order matches the order in the JSONL file
    3. Ground truth from step file matches answer from JSONL file

    Args:
        step_file_path: Path to the step JSON file
        jsonl_file_path: Path to the JSONL file

    Returns:
        List of merged dictionaries combining data from both files

    Raises:
        ValueError: If ground truth doesn't match answer field
    """
    # Load both files
    step_data = load_step_file(step_file_path)
    jsonl_data = load_jsonl_file(jsonl_file_path)

    # Group step data by dataset
    step_grouped = group_step_data_by_dataset(step_data)

    # Prepare result list
    merged_data = []

    # Track position in each dataset's samples from step file
    dataset_indices = defaultdict(int)

    # Iterate through JSONL data in order
    for jsonl_sample in jsonl_data:
        dataset = jsonl_sample['dataset']

        # Get the corresponding sample from step data
        if dataset not in step_grouped:
            raise ValueError(
                f"Dataset '{dataset}' found in JSONL file but not in step file"
            )

        idx = dataset_indices[dataset]

        if idx >= len(step_grouped[dataset]):
            raise ValueError(
                f"Not enough samples for dataset '{dataset}' in step file. "
                f"Expected at least {idx + 1} samples, got {len(step_grouped[dataset])}"
            )

        step_sample = step_grouped[dataset][idx]

        # Verify ground truth matches answer
        if step_sample['ground_truth'] != jsonl_sample['answer']:
            raise ValueError(
                f"Ground truth mismatch at dataset '{dataset}', index {idx}:\n"
                f"  Step file ground_truth: '{step_sample['ground_truth']}'\n"
                f"  JSONL file answer: '{jsonl_sample['answer']}'\n"
                f"  Original step file index: {step_sample['original_index']}"
            )

        # Merge the data
        merged_sample = {
            **jsonl_sample,  # Include all JSONL fields
            'prediction': step_sample['prediction'],
            'step_ground_truth': step_sample['ground_truth'],
            'step_file_index': step_sample['original_index']
        }

        merged_data.append(merged_sample)

        # Move to next sample for this dataset
        dataset_indices[dataset] += 1

    # Verify all samples from step file were used
    for dataset, samples in step_grouped.items():
        used_count = dataset_indices[dataset]
        total_count = len(samples)
        if used_count != total_count:
            raise ValueError(
                f"Dataset '{dataset}' has {total_count} samples in step file "
                f"but only {used_count} samples in JSONL file"
            )

    return merged_data


def save_merged_data(merged_data: List[Dict[str, Any]], output_path: str):
    """
    Save merged data to a JSONL file.

    Args:
        merged_data: List of merged dictionaries
        output_path: Path to save the output JSONL file
    """
    with open(output_path, 'w') as f:
        for sample in merged_data:
            f.write(json.dumps(sample) + '\n')


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python compare_reasoning.py <step_file> <jsonl_file> [output_file]")
        print("\nExample:")
        print("  python compare_reasoning.py step_500.json final_v8_val_cleaned.jsonl merged_output.jsonl")
        sys.exit(1)

    step_file = sys.argv[1]
    jsonl_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "merged_output.jsonl"

    try:
        print(f"Loading step file: {step_file}")
        print(f"Loading JSONL file: {jsonl_file}")

        merged_data = merge_files(step_file, jsonl_file)

        print(f"\nSuccessfully merged {len(merged_data)} samples!")
        print(f"Saving to: {output_file}")

        save_merged_data(merged_data, output_file)

        print("Done!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
