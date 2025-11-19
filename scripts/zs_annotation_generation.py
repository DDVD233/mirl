import json
import os
from collections import defaultdict


def load_jsonl(filepath):
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data, filepath):
    """Save a list of dictionaries to a JSONL file."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def create_zero_shot_split(train_data, test_data, target_datasets):
    """
    Create zero-shot train/test split while maintaining original train/test separation.

    Args:
        train_data: List of training samples from v5_train_upd.jsonl
        test_data: List of testing samples from v5_test_upd.jsonl
        target_datasets: List of dataset names for zero-shot testing

    Returns:
        Tuple of (zs_train_data, zs_test_data)
    """
    # Group TRAIN data by dataset
    train_by_dataset = defaultdict(list)
    for sample in train_data:
        dataset_name = sample.get('dataset', 'unknown')
        train_by_dataset[dataset_name].append(sample)

    # Group TEST data by dataset
    test_by_dataset = defaultdict(list)
    for sample in test_data:
        dataset_name = sample.get('dataset', 'unknown')
        test_by_dataset[dataset_name].append(sample)

    # Print statistics about available datasets
    print("Available datasets in TRAIN:")
    for dataset, samples in sorted(train_by_dataset.items()):
        print(f"  {dataset}: {len(samples)} samples")
    print()

    print("Available datasets in TEST:")
    for dataset, samples in sorted(test_by_dataset.items()):
        print(f"  {dataset}: {len(samples)} samples")
    print()

    # Create zero-shot test set (only target datasets from TEST data)
    zs_test = []
    test_datasets_found = []
    for target_dataset in target_datasets:
        if target_dataset in test_by_dataset:
            zs_test.extend(test_by_dataset[target_dataset])
            test_datasets_found.append(target_dataset)
        else:
            print(f"Warning: Target dataset '{target_dataset}' not found in TEST data.")

    # Create zero-shot train set (everything from TRAIN except target datasets)
    zs_train = []
    train_datasets_kept = []
    for dataset, samples in train_by_dataset.items():
        if dataset not in target_datasets:
            zs_train.extend(samples)
            train_datasets_kept.append(dataset)

    # Print split statistics
    print("\nZero-shot split created:")
    print(f"Test set (from original test, only zero-shot datasets): {len(zs_test)} samples")
    if test_datasets_found:
        print(f"  Datasets included: {', '.join(test_datasets_found)}")
    else:
        print(f"  No target datasets found in test set!")

    print(f"Train set (from original train, excluding zero-shot datasets): {len(zs_train)} samples")
    if train_datasets_kept:
        print(f"  Datasets included: {', '.join(sorted(train_datasets_kept))}")
    else:
        print(f"  No non-target datasets found in train set!")

    return zs_train, zs_test


def main():
    # Input files
    train_file = 'v5_train_upd.jsonl'
    test_file = 'v5_test_upd.jsonl'

    # Target datasets for zero-shot testing
    target_datasets = ['mosei_senti', 'meld_emotion', 'daicwoz', 'mmsd']

    # Check if input files exist
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found!")
        return
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found!")
        return

    # Load data
    print(f"Loading {train_file}...")
    train_data = load_jsonl(train_file)
    print(f"Loaded {len(train_data)} training samples")

    print(f"Loading {test_file}...")
    test_data = load_jsonl(test_file)
    print(f"Loaded {len(test_data)} testing samples")
    print()

    # Create zero-shot split
    zs_train, zs_test = create_zero_shot_split(train_data, test_data, target_datasets)

    # Output files
    output_train = 'zs_train.jsonl'
    output_test = 'zs_test.jsonl'

    # Save the splits
    print(f"\nSaving files...")
    save_jsonl(zs_train, output_train)
    print(f"  Saved training set: {output_train} ({len(zs_train)} samples)")

    save_jsonl(zs_test, output_test)
    print(f"  Saved test set: {output_test} ({len(zs_test)} samples)")

    print("\nZero-shot dataset creation complete!")


if __name__ == "__main__":
    main()