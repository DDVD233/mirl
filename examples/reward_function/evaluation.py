import datetime
import json
import os
from collections import defaultdict
from typing import Dict, List, Set
import statistics

def parse_conditions(text: str) -> Set[str]:
    """
    Parse medical conditions from text, handling various separators.

    Args:
        text (str): Text containing medical conditions.

    Returns:
        Set[str]: Set of individual medical conditions.
    """
    # Remove any boxing notation if present
    text = text.replace("\\boxed{", "").replace("}", "")

    # Split by common separators
    for sep in [", ", " and ", " & ", ",", "&"]:
        if sep in text:
            return set(cond.strip() for cond in text.split(sep))

    # If no separator found, treat as single condition
    return {text.strip()}


def extract_boxed_content(text: str) -> str:
    """
    Extract content within \boxed{} or similar boxing notations.

    Args:
        text (str): Text containing potentially boxed content.

    Returns:
        str: Extracted boxed content or the original text if no box found.
    """
    import re

    # Look for LaTeX \boxed{} notation
    boxed_match = re.search(r"\\boxed{([^}]*)}", text)
    if boxed_match:
        return boxed_match.group(1)

    # Look for markdown boxed notation (e.g., [boxed content])
    markdown_match = re.search(r"\[(.*?)\]", text)
    if markdown_match:
        return markdown_match.group(1)

    # Return the text as is if no boxed content is found
    return text


def compute_class_metrics(class_name: str, confusion_matrix: Dict[str, int]) -> Dict[str, float]:
    """
    Compute metrics for a single class based on its confusion matrix.

    Args:
        class_name (str): Name of the class.
        confusion_matrix (Dict[str, int]): Confusion matrix with tp, fp, fn, tn.

    Returns:
        Dict[str, float]: Dictionary of metrics for this class.
    """
    tp = confusion_matrix["tp"]
    fp = confusion_matrix["fp"]
    fn = confusion_matrix["fn"]
    tn = confusion_matrix["tn"]

    def divide_by_zero(n, d):
        return n / d if d else 0.0

    precision = divide_by_zero(tp, tp + fp)
    recall = divide_by_zero(tp, tp + fn)
    sensitivity = recall
    specificity = divide_by_zero(tn, tn + fp)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = divide_by_zero(tp + tn, tp + tn + fp + fn)
    tpr = sensitivity
    fpr = divide_by_zero(fp, fp + tn)
    fdr = divide_by_zero(fp, tp + fp)
    return {
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "fdr": fdr,
        "count": confusion_matrix["count"],
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def gender(predictions: List[str], ground_truths: List[str], demographics: List[str]) -> Dict[str, float]:
    groups = {"male": {"preds": [], "gts": []}, "female": {"preds": [], "gts": []}}

    for pred, gt, demo in zip(predictions, ground_truths, demographics):
        if demo is not None and "female" in demo.lower():
            groups["female"]["preds"].append(pred)
            groups["female"]["gts"].append(gt)
        elif demo is not None and "male" in demo.lower():
            groups["male"]["preds"].append(pred)
            groups["male"]["gts"].append(gt)

    results = {}
    acc_values = []
    f1_values = []
    tpr_values = []
    fpr_values = []
    fdr_values = []

    for sex in ["male", "female"]:
        preds = groups[sex]["preds"]
        gts = groups[sex]["gts"]
        if len(preds) == 0:
            continue
        metrics = compute_dataset_metrics(preds, gts)["dataset_metrics"]
        acc = metrics["accuracy"]
        f1 = metrics["f1"]
        tpr = metrics.get("tpr", metrics["sensitivity"])
        fdr = metrics.get("fdr", 1 - metrics["precision"])
        fpr = metrics.get("fpr", 1 - metrics["specificity"])

        results[f"{sex}/accuracy"] = acc
        results[f"{sex}/f1"] = f1
        results[f"{sex}/tpr"] = tpr
        results[f"{sex}/fdr"] = fdr
        results[f"{sex}/fpr"] = fpr

        acc_values.append(acc)
        f1_values.append(f1)
        tpr_values.append(tpr)
        fdr_values.append(fdr)
        fpr_values.append(fpr)

    print(f"{sex}: accuracy = {acc:.4f}, f1 = {f1:.4f}, tpr = {tpr:.4f}, fdr = {fdr:.4f}")

    if len(acc_values) >= 2:
        acc_diff = abs(acc_values[0] - acc_values[1])
        results["acc_diff for sex"] = acc_diff
        results["std_accuracy for sex"] = statistics.stdev(acc_values)
        print(f"Accuracy max diff for sex = {acc_diff:.4f}")
        print(f"std of accuracy for sex = {results['std_accuracy for sex']:.4f}")

    if len(f1_values) >= 2:
        f1_diff = abs(f1_values[0] - f1_values[1])
        results["f1_diff for sex"] = f1_diff
        results["std_f1 for sex"] = statistics.stdev(f1_values)
        print(f"F1 max diff for sex = {f1_diff:.4f}")
        print(f"std of f1 for sex = {results['std_f1 for sex']:.4f}")

    if len(tpr_values) >= 2:
        results["tpr_diff for sex"] = abs(tpr_values[0] - tpr_values[1])
        results["std_tpr for sex"] = statistics.stdev(tpr_values)
        print(f"TPR max diff for sex = {results['tpr_diff for sex']:.4f}")
        print(f"std of tpr for sex = {results['std_tpr for sex']:.4f}")

    if len(fdr_values) >= 2:
        results["fdr_diff for sex"] = abs(fdr_values[0] - fdr_values[1])
        results["std_fdr for sex"] = statistics.stdev(fdr_values)
        print(f"FDR max diff for sex = {results['fdr_diff for sex']:.4f}")
        print(f"std of fdr for sex = {results['std_fdr for sex']:.4f}")

    if len(fpr_values) >= 2:
        results["fpr_diff for sex"] = abs(fpr_values[0] - fpr_values[1])
        results["std_fpr for sex"] = statistics.stdev(fpr_values)


    return results


def parent(predictions: List[str], ground_truths: List[str], demographics: List[str]) -> Dict[str, float]:
    groups = {}
    for pred, gt, demo in zip(predictions, ground_truths, demographics):
        if demo is not None and "father" in demo.lower():
            if (
                demo.split("father:")[1].strip().split()[0] not in groups
                and demo.split("father:")[1].strip().split()[0] != "NAN"
            ):
                groups[demo.split("father:")[1].strip().split()[0]] = {"preds": [], "gts": []}
                groups[demo.split("father:")[1].strip().split()[0]]["preds"].append(pred)
                groups[demo.split("father:")[1].strip().split()[0]]["gts"].append(gt)
            else:
                groups[demo.split("father:")[1].strip().split()[0]]["preds"].append(pred)
                groups[demo.split("father:")[1].strip().split()[0]]["gts"].append(gt)
        if demo is not None and "mother" in demo.lower():
            if (
                demo.split("mother:")[1].strip().split()[0] not in groups
                and demo.split("mother:")[1].strip().split()[0] != "NAN"
            ):
                groups[demo.split("mother:")[1].strip().split()[0]] = {"preds": [], "gts": []}
                groups[demo.split("mother:")[1].strip().split()[0]]["preds"].append(pred)
                groups[demo.split("mother:")[1].strip().split()[0]]["gts"].append(gt)
            else:
                groups[demo.split("father:")[1].strip().split()[0]]["preds"].append(pred)
                groups[demo.split("father:")[1].strip().split()[0]]["gts"].append(gt)

    results = {}
    acc_values = []
    f1_values = []
    tpr_values = []
    fpr_values = []
    fdr_values = []

    for race in groups:
        preds = groups[race]["preds"]
        gts = groups[race]["gts"]
        if len(preds) == 0:
            continue
        metrics = compute_dataset_metrics(preds, gts)["dataset_metrics"]
        acc = metrics["accuracy"]
        f1 = metrics["f1"]
        tpr = metrics.get("tpr", metrics["sensitivity"])
        fpr = metrics.get("fpr", 1.0 - metrics["specificity"])
        fdr = metrics.get("fdr", 1.0 - metrics["precision"])

        results[f"{race}/accuracy"] = acc
        results[f"{race}/f1"] = f1
        results[f"{race}/tpr"] = tpr
        results[f"{race}/fpr"] = fpr
        results[f"{race}/fdr"] = fdr

        acc_values.append(acc)
        f1_values.append(f1)
        tpr_values.append(tpr)
        fpr_values.append(fpr)
        fdr_values.append(fdr)
        print(f"{race}: accuracy = {acc:.4f}, f1 = {f1:.4f}, tpr = {tpr:.4f}, fpr = {fpr:.4f}, fdr = {fdr:.4f}")

    if len(acc_values) >= 2:
        acc_diff = max(acc_values) - min(acc_values)
        results["acc_diff"] = acc_diff
        print(f"Accuracy max diff for parent = {acc_diff:.4f}")
        std_acc = statistics.stdev(acc_values)
        results["std_accuracy"] = std_acc
        print(f"std of accuracy for parent = {std_acc:.4f}")

    if len(f1_values) >= 2:
        f1_diff = max(f1_values) - min(f1_values)
        results["f1_diff"] = f1_diff
        print(f"F1 max diff for parent = {f1_diff:.4f}")
        std_f1 = statistics.stdev(f1_values)
        results["std_f1"] = std_f1
        print(f"std of f1 for parent = {std_f1:.4f}")

    if len(tpr_values) >= 2:
        results["tpr_diff for parent"] = max(tpr_values) - min(tpr_values)
        results["std_tpr for parent"] = statistics.stdev(tpr_values)
        print(f"TPR max diff for parent = {results['tpr_diff for parent']:.4f}")
        print(f"std of tpr for parent = {results['std_tpr for parent']:.4f}")

    if len(fpr_values) >= 2:
        results["fpr_diff for parent"] = max(fpr_values) - min(fpr_values)
        results["std_fpr for parent"] = statistics.stdev(fpr_values)
        print(f"FPR max diff for parent = {results['fpr_diff for parent']:.4f}")
        print(f"std of fpr for parent = {results['std_fpr for parent']:.4f}")

    if len(fdr_values) >= 2:
        results["fdr_diff for parent"] = max(fdr_values) - min(fdr_values)
        results["std_fdr for parent"] = statistics.stdev(fdr_values)
        print(f"FDR max diff for parent = {results['fdr_diff for parent']:.4f}")
        print(f"std of fdr for parent = {results['std_fdr for parent']:.4f}")

    return results


def age(predictions: List[str], ground_truths: List[str], demographics: List[str]) -> Dict[str, float]:
    groups = {
        "a1": {"preds": [], "gts": []},
        "a2": {"preds": [], "gts": []},
        "a3": {"preds": [], "gts": []},
        "a4": {"preds": [], "gts": []},
    }

    for pred, gt, demo in zip(predictions, ground_truths, demographics):
        if demo is not None and "age" in demo.lower():
            try:
                age_str = demo.split("age:")[1].strip().split()[0].replace(",", "")
                age_val = float(age_str)
            except (IndexError, ValueError):
                continue

            if age_val <= 25:
                groups["a1"]["preds"].append(pred)
                groups["a1"]["gts"].append(gt)
            elif 25 < age_val <= 50:
                groups["a2"]["preds"].append(pred)
                groups["a2"]["gts"].append(gt)
            elif 50 < age_val <= 75:
                groups["a3"]["preds"].append(pred)
                groups["a3"]["gts"].append(gt)
            elif 75 < age_val:
                groups["a4"]["preds"].append(pred)
                groups["a4"]["gts"].append(gt)

    results = {}
    acc_values = []
    f1_values = []
    tpr_values = []
    fpr_values = []
    fdr_values = []

    for group in ["a1", "a2", "a3", "a4"]:
        preds = groups[group]["preds"]
        gts = groups[group]["gts"]
        if len(preds) == 0:
            continue
        metrics = compute_dataset_metrics(preds, gts)["dataset_metrics"]
        acc = metrics["accuracy"]
        f1 = metrics["f1"]
        tpr = metrics.get("tpr", metrics["sensitivity"])
        fpr = metrics.get("fpr", 1.0 - metrics["specificity"])
        fdr = metrics.get("fdr", 1.0 - metrics["precision"])

        results[f"{group}/accuracy"] = acc
        results[f"{group}/f1"] = f1
        results[f"{group}/tpr"] = tpr
        results[f"{group}/fpr"] = fpr
        results[f"{group}/fdr"] = fdr

        acc_values.append(acc)
        f1_values.append(f1)
        tpr_values.append(tpr)
        fpr_values.append(fpr)
        fdr_values.append(fdr)

    if len(acc_values) >= 2:
        results["acc_diff"] = max(acc_values) - min(acc_values)
        results["std_accuracy"] = statistics.stdev(acc_values)

    if len(f1_values) >= 2:
        results["f1_diff"] = max(f1_values) - min(f1_values)
        results["std_f1"] = statistics.stdev(f1_values)

    for group in ["a1", "a2", "a3", "a4"]:
        acc = results.get(f"{group}/accuracy")
        f1 = results.get(f"{group}/f1")
        if acc is not None and f1 is not None:
            print(f"{group}: accuracy = {acc:.4f}, f1 = {f1:.4f}")

    if "acc_diff" in results:
        results["acc_diff"] = max(acc_values) - min(acc_values)
        results["std_accuracy"] = statistics.stdev(acc_values)
        print(f"Accuracy max diff = {results['acc_diff']:.4f}")
        print(f"std of accuracy for age = {results['std_accuracy']:.4f}")
    if "f1_diff" in results:
        results["f1_diff"] = max(f1_values) - min(f1_values)
        results["std_f1"] = statistics.stdev(f1_values)
        print(f"F1 max diff = {results['f1_diff']:.4f}")
        print(f"std of f1 for age = {results['std_f1']:.4f}")
    if len(tpr_values) >= 2:
        results["tpr_diff"] = max(tpr_values) - min(tpr_values)
        results["std_tpr"] = statistics.stdev(tpr_values)
        print(f"TPR max diff = {results['tpr_diff']:.4f}")
        print(f"std of tpr for age = {results['std_tpr']:.4f}")
    if len(fpr_values) >= 2:
        results["fpr_diff"] = max(fpr_values) - min(fpr_values)
        results["std_fpr"] = statistics.stdev(fpr_values)
        print(f"FPR max diff = {results['fpr_diff']:.4f}")
        print(f"std of fpr for age = {results['std_fpr']:.4f}")
    if len(fdr_values) >= 2:
        results["fdr_diff"] = max(fdr_values) - min(fdr_values)
        results["std_fdr"] = statistics.stdev(fdr_values)
        print(f"FDR max diff = {results['fdr_diff']:.4f}")
        print(f"std of fdr for age = {results['std_fdr']:.4f}")

    return results
def compute_confusion_matrices(predictions: List[str], ground_truths: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Compute confusion matrices for each class.

    Args:
        predictions (List[str]): List of model predictions.
        ground_truths (List[str]): List of ground truth labels.

    Returns:
        Dict[str, Dict[str, int]]: Confusion matrices for each class.
    """
    # Initialize counters for each condition
    all_conditions = set()
    condition_matrices = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "count": 0})

    # First pass: identify all unique conditions
    for gt in ground_truths:
        gt_conditions = parse_conditions(gt)
        all_conditions.update(gt_conditions)

    for pred in predictions:
        pred_answer = extract_boxed_content(pred)
        if pred_answer != "None":
            pred_conditions = parse_conditions(pred_answer)
            all_conditions.update(pred_conditions)

    # Second pass: compute confusion matrices
    for pred, gt in zip(predictions, ground_truths):
        pred_answer = extract_boxed_content(pred)
        if pred_answer == "None":
            pred_conditions = set()
        else:
            pred_conditions = parse_conditions(pred_answer)

        gt_conditions = parse_conditions(gt)

        # For each possible condition
        for condition in all_conditions:
            condition_present_in_gt = condition in gt_conditions
            condition_present_in_pred = condition in pred_conditions

            if condition_present_in_gt:
                condition_matrices[condition]["count"] += 1

            if condition_present_in_gt and condition_present_in_pred:
                # True positive
                condition_matrices[condition]["tp"] += 1
            elif condition_present_in_gt and not condition_present_in_pred:
                # False negative
                condition_matrices[condition]["fn"] += 1
            elif not condition_present_in_gt and condition_present_in_pred:
                # False positive
                condition_matrices[condition]["fp"] += 1
            else:
                # True negative
                condition_matrices[condition]["tn"] += 1

    return condition_matrices


def compute_dataset_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, Dict]:
    """
    Compute metrics for a single dataset, with class-wise averaging.

    Args:
        predictions (List[str]): List of model predictions for this dataset.
        ground_truths (List[str]): List of ground truth labels for this dataset.

    Returns:
        Dict[str, Dict]: Class metrics and averaged dataset metrics.
    """
    # Compute confusion matrices for each class
    class_matrices = compute_confusion_matrices(predictions, ground_truths)

    # Compute metrics for each class
    class_metrics = {}
    active_classes = 0

    # Accumulators for dataset-level metrics
    dataset_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
        "tpr": 0.0,
        "fpr": 0.0,
        "fdr": 0.0,
    }

    # Compute metrics for each class and accumulate for dataset average
    for class_name, matrix in class_matrices.items():
        # Skip classes that never appear in ground truth
        if matrix["count"] == 0:
            continue

        active_classes += 1
        metrics = compute_class_metrics(class_name, matrix)
        class_metrics[class_name] = metrics

        # Accumulate for dataset average (equal class weighting)
        for metric_name in dataset_metrics.keys():
            dataset_metrics[metric_name] += metrics[metric_name]

    # Calculate dataset average (equal class weighting)
    if active_classes > 0:
        for metric_name in dataset_metrics.keys():
            dataset_metrics[metric_name] /= active_classes

    # Add class metrics to the result
    result = {"class_metrics": class_metrics, "dataset_metrics": dataset_metrics, "active_classes": active_classes}

    return result


def compute_metrics_by_data_source(
    predictions: List[str],
    ground_truths: List[str],
    data_sources: List[str],
    datasets: List[str],
    demographics: List[str],
) -> Dict[str, float]:
    """
    Compute hierarchical metrics: class -> dataset -> data source -> global.

    Args:
        predictions (List[str]): List of model predictions.
        ground_truths (List[str]): List of ground truth labels.
        data_sources (List[str]): List of data sources for each example.
        datasets (List[str]): List of dataset identifiers for each example.
        demographics (List[str]): List of demographic information for each example.

    Returns:
        Dict[str, float]: Flattened dictionary of metrics at all levels with keys:
            - "val/{metric}" for global metrics
            - "{data_source}/{metric}" for data source metrics
            - "{data_source}/{dataset}/{metric}" for dataset metrics
    """
    # Save inputs to json for debugging under outputs/

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    input_data = {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "data_sources": data_sources,
        "datasets": datasets,
        "demographics": demographics,
    }
    # name is time in yyyy-mm-dd_hh-mm-ss format
    with open(
        os.path.join(output_dir, f"input_data_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"), "w"
    ) as f:
        json.dump(input_data, f, indent=4)

    # Group examples by data source and dataset
    grouped_data = defaultdict(lambda: defaultdict(lambda: {"preds": [], "gts": []}))

    for pred, gt, source, dataset in zip(predictions, ground_truths, data_sources, datasets):
        grouped_data[source][dataset]["preds"].append(pred)
        grouped_data[source][dataset]["gts"].append(gt)

    # Initialize the flattened result dictionary
    result = {}

    # Initialize global metrics accumulators
    global_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
    }

    # Compute metrics for each dataset within each data source
    total_data_sources = 0

    for source_name, source_datasets in grouped_data.items():
        # Initialize metrics accumulators for this data source
        source_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

        total_datasets_in_source = 0

        for dataset_name, dataset_data in source_datasets.items():
            # Compute metrics for this dataset
            dataset_result = compute_dataset_metrics(dataset_data["preds"], dataset_data["gts"])

            # Store dataset-level metrics with the format "data_source/dataset/metric"
            for metric_name, metric_value in dataset_result["dataset_metrics"].items():
                result[f"{source_name}/{dataset_name}/{metric_name}"] = metric_value

            # Skip empty datasets
            if dataset_result["active_classes"] == 0:
                continue

            total_datasets_in_source += 1

            # Accumulate metrics for data source average (equal dataset weighting)
            for metric_name in source_metrics.keys():
                source_metrics[metric_name] += dataset_result["dataset_metrics"][metric_name]

        # Calculate data source average (equal dataset weighting)
        if total_datasets_in_source > 0:
            for metric_name in source_metrics.keys():
                source_metrics[metric_name] /= total_datasets_in_source

            # Store data source metrics with the format "data_source/metric"
            for metric_name, metric_value in source_metrics.items():
                result[f"{source_name}/{metric_name}"] = metric_value

            total_data_sources += 1

            # Accumulate for global metrics (equal data source weighting)
            for metric_name in global_metrics.keys():
                global_metrics[metric_name] += source_metrics[metric_name]

    # Calculate global average (equal data source weighting)
    if total_data_sources > 0:
        for metric_name in global_metrics.keys():
            global_metrics[metric_name] /= total_data_sources

        # Store global metrics with the format "val/metric"
        for metric_name, metric_value in global_metrics.items():
            result[f"val/{metric_name}"] = metric_value

    gender_results = gender(predictions, ground_truths, demographics)
    for k, v in gender_results.items():
        result[f"fairness/gender/{k}"] = v

    age_results = age(predictions, ground_truths, demographics)
    for k, v in age_results.items():
        result[f"fairness/age/{k}"] = v

    parent_results = parent(predictions, ground_truths, demographics)
    for k, v in parent_results.items():
        result[f"fairness/parent/{k}"] = v


    std_acc_values = []
    std_f1_values = []
    try:

        std_acc_values.append(gender_results["std_accuracy for sex"])
        std_f1_values.append(gender_results["std_f1 for sex"])


        std_acc_values.append(age_results["std_accuracy"])
        std_f1_values.append(age_results["std_f1"])

        std_acc_values.append(parent_results["std_accuracy"])
        std_f1_values.append(parent_results["std_f1"])

        result["fairness/avg_std_accuracy"] = sum(std_acc_values) / len(std_acc_values)
        result["fairness/avg_std_f1"] = sum(std_f1_values) / len(std_f1_values)
    except KeyError:
        print("Some fairness metrics do not have standard deviation values, skipping average calculation.")

    return result


if __name__ == "__main__":
    outputs_dir = "../../outputs"
    output_files = [f for f in os.listdir(outputs_dir) if f.startswith("input_data_") and f.endswith(".json")]
    if not output_files:
        print("No output files found in the outputs directory.")
    else:
        latest_file = max(output_files, key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)))
        with open(os.path.join(outputs_dir, latest_file), "r") as f:
            input_data = json.load(f)

        predictions = input_data["predictions"]
        ground_truths = input_data["ground_truths"]
        data_sources = input_data["data_sources"]
        datasets = input_data["datasets"]
        demographics = input_data["demographics"]

        metrics = compute_metrics_by_data_source(predictions, ground_truths, data_sources, datasets, demographics)
        print(json.dumps(metrics, indent=4))