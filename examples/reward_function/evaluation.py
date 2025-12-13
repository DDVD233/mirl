import datetime
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Set
import statistics


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags and return the content after."""
    # Remove everything within <think>...</think> tags
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def jaccard_similarity(pred_label: str, ground_truth: str) -> float:
    """
    Compute Jaccard similarity (token overlap) between predicted label and ground truth.

    Args:
        pred_label: Predicted label string
        ground_truth: Ground truth string

    Returns:
        Jaccard similarity score between 0 and 1
    """
    # Strip thinking tags from prediction
    pred_label = strip_thinking_tags(pred_label)

    pred_tokens = set(pred_label.lower().split())
    gt_tokens = set(ground_truth.lower().split())

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    intersection = len(pred_tokens & gt_tokens)
    union = len(pred_tokens | gt_tokens)
    return intersection / union if union > 0 else 0.0


def compute_pairwise_similarities(predictions: List[str], ground_truths: List[str]) -> List[float]:
    """
    Compute Jaccard similarity for each prediction-ground_truth pair.
    This is computed once and reused for all aggregations.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth labels

    Returns:
        List of similarity scores for each pair
    """
    similarities = []
    for pred, gt in zip(predictions, ground_truths):
        pred_answer = extract_boxed_content(pred)
        if pred_answer == "None" or pred_answer == "":
            similarities.append(0.0)
        else:
            similarities.append(jaccard_similarity(pred_answer, gt))
    return similarities

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


def gender(predictions: List[str], ground_truths: List[str], demographics: List[str], similarities: List[float] = None) -> Dict[str, float]:
    groups = {"male": {"preds": [], "gts": [], "sims": []}, "female": {"preds": [], "gts": [], "sims": []}}

    # Compute similarities if not provided
    if similarities is None:
        similarities = compute_pairwise_similarities(predictions, ground_truths)

    for pred, gt, demo, sim in zip(predictions, ground_truths, demographics, similarities):
        if demo is not None and "female" in demo.lower():
            groups["female"]["preds"].append(pred)
            groups["female"]["gts"].append(gt)
            groups["female"]["sims"].append(sim)
        elif demo is not None and "male" in demo.lower():
            groups["male"]["preds"].append(pred)
            groups["male"]["gts"].append(gt)
            groups["male"]["sims"].append(sim)

    results = {}
    acc_values = []
    f1_values = []
    sim_values = []
    tpr_values = []
    fpr_values = []
    fdr_values = []

    for sex in ["male", "female"]:
        preds = groups[sex]["preds"]
        gts = groups[sex]["gts"]
        sims = groups[sex]["sims"]
        if len(preds) == 0:
            continue
        metrics = compute_dataset_metrics(preds, gts, sims)["dataset_metrics"]
        acc = metrics["accuracy"]
        f1 = metrics["f1"]
        sim = metrics["similarity"]
        tpr = metrics.get("tpr", metrics["sensitivity"])
        fdr = metrics.get("fdr", 1 - metrics["precision"])
        fpr = metrics.get("fpr", 1 - metrics["specificity"])

        results[f"{sex}/accuracy"] = acc
        results[f"{sex}/f1"] = f1
        results[f"{sex}/similarity"] = sim
        results[f"{sex}/tpr"] = tpr
        results[f"{sex}/fdr"] = fdr
        results[f"{sex}/fpr"] = fpr

        acc_values.append(acc)
        f1_values.append(f1)
        sim_values.append(sim)
        tpr_values.append(tpr)
        fdr_values.append(fdr)
        fpr_values.append(fpr)

    results["acc"] = sum(acc_values) / len(acc_values)
    results["f1"] = sum(f1_values) / len(f1_values)
    results["similarity"] = sum(sim_values) / len(sim_values)
    results["tpr"] = sum(tpr_values) / len(tpr_values)
    results["fpr"] = sum(fpr_values) / len(fpr_values)
    results["fdr"] = sum(fdr_values) / len(fdr_values)

    print(f"{sex}: accuracy = {acc:.4f}, f1 = {f1:.4f}, similarity = {sim:.4f}, tpr = {tpr:.4f}, fdr = {fdr:.4f}")

    if len(acc_values) >= 2:
        acc_diff = abs(acc_values[0] - acc_values[1])
        results["acc_diff"] = acc_diff
        results["acc_std"] = statistics.stdev(acc_values)
        print(f"Accuracy max diff = {acc_diff:.4f}")
        print(f"std of accuracy = {results['acc_std']:.4f}")

    if len(f1_values) >= 2:
        f1_diff = abs(f1_values[0] - f1_values[1])
        results["f1_diff"] = f1_diff
        results["f1_std"] = statistics.stdev(f1_values)
        print(f"F1 max diff = {f1_diff:.4f}")
        print(f"std of f1 = {results['f1_std']:.4f}")

    if len(sim_values) >= 2:
        sim_diff = abs(sim_values[0] - sim_values[1])
        results["similarity_diff"] = sim_diff
        results["similarity_std"] = statistics.stdev(sim_values)
        print(f"Similarity max diff = {sim_diff:.4f}")
        print(f"std of similarity = {results['similarity_std']:.4f}")

    if len(tpr_values) >= 2:
        results["tpr_diff"] = abs(tpr_values[0] - tpr_values[1])
        results["std_tpr"] = statistics.stdev(tpr_values)
        print(f"TPR max diff = {results['tpr_diff']:.4f}")
        print(f"std of tpr = {results['std_tpr']:.4f}")

    if len(fdr_values) >= 2:
        results["fdr_diff"] = abs(fdr_values[0] - fdr_values[1])
        results["std_fdr"] = statistics.stdev(fdr_values)
        print(f"FDR max diff = {results['fdr_diff']:.4f}")
        print(f"std of fdr = {results['std_fdr']:.4f}")

    if len(fpr_values) >= 2:
        results["fpr_diff"] = abs(fpr_values[0] - fpr_values[1])
        results["std_fpr"] = statistics.stdev(fpr_values)


    return results


def parent(predictions: List[str], ground_truths: List[str], demographics: List[str], similarities: List[float] = None) -> Dict[str, float]:
    groups = {}

    # Compute similarities if not provided
    if similarities is None:
        similarities = compute_pairwise_similarities(predictions, ground_truths)

    for pred, gt, demo, sim in zip(predictions, ground_truths, demographics, similarities):
        if demo is not None and "father" in demo.lower():
            key = demo.split("father:")[1].strip().split()[0]
            if key not in groups and key != "NAN":
                groups[key] = {"preds": [], "gts": [], "sims": []}
            if key in groups:
                groups[key]["preds"].append(pred)
                groups[key]["gts"].append(gt)
                groups[key]["sims"].append(sim)
        if demo is not None and "mother" in demo.lower():
            key = demo.split("mother:")[1].strip().split()[0]
            if key not in groups and key != "NAN":
                groups[key] = {"preds": [], "gts": [], "sims": []}
            if key in groups:
                groups[key]["preds"].append(pred)
                groups[key]["gts"].append(gt)
                groups[key]["sims"].append(sim)

    results = {}
    acc_values = []
    f1_values = []
    sim_values = []
    tpr_values = []
    fpr_values = []
    fdr_values = []

    for race in groups:
        preds = groups[race]["preds"]
        gts = groups[race]["gts"]
        sims = groups[race]["sims"]
        if len(preds) == 0:
            continue
        metrics = compute_dataset_metrics(preds, gts, sims)["dataset_metrics"]
        acc = metrics["accuracy"]
        f1 = metrics["f1"]
        sim = metrics["similarity"]
        tpr = metrics.get("tpr", metrics["sensitivity"])
        fpr = metrics.get("fpr", 1.0 - metrics["specificity"])
        fdr = metrics.get("fdr", 1.0 - metrics["precision"])

        results[f"{race}/accuracy"] = acc
        results[f"{race}/f1"] = f1
        results[f"{race}/similarity"] = sim
        results[f"{race}/tpr"] = tpr
        results[f"{race}/fpr"] = fpr
        results[f"{race}/fdr"] = fdr

        acc_values.append(acc)
        f1_values.append(f1)
        sim_values.append(sim)
        tpr_values.append(tpr)
        fpr_values.append(fpr)
        fdr_values.append(fdr)
        print(f"{race}: accuracy = {acc:.4f}, f1 = {f1:.4f}, similarity = {sim:.4f}, tpr = {tpr:.4f}, fpr = {fpr:.4f}, fdr = {fdr:.4f}")

    results["acc"] = sum(acc_values) / len(acc_values)
    results["f1"] = sum(f1_values) / len(f1_values)
    results["similarity"] = sum(sim_values) / len(sim_values)
    results["tpr"] = sum(tpr_values) / len(tpr_values)
    results["fpr"] = sum(fpr_values) / len(fpr_values)
    results["fdr"] = sum(fdr_values) / len(fdr_values)

    if len(acc_values) >= 2:
        acc_diff = max(acc_values) - min(acc_values)
        results["acc_diff"] = acc_diff
        print(f"Accuracy max diff for parent = {acc_diff:.4f}")
        std_acc = statistics.stdev(acc_values)
        results["acc_std"] = std_acc
        print(f"std of accuracy for parent = {std_acc:.4f}")

    if len(f1_values) >= 2:
        f1_diff = max(f1_values) - min(f1_values)
        results["f1_diff"] = f1_diff
        print(f"F1 max diff for parent = {f1_diff:.4f}")
        f1_std = statistics.stdev(f1_values)
        results["f1_std"] = f1_std
        print(f"std of f1 for parent = {f1_std:.4f}")

    if len(sim_values) >= 2:
        sim_diff = max(sim_values) - min(sim_values)
        results["similarity_diff for parent"] = sim_diff
        results["similarity_std for parent"] = statistics.stdev(sim_values)
        print(f"Similarity max diff for parent = {sim_diff:.4f}")
        print(f"std of similarity for parent = {results['similarity_std for parent']:.4f}")

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


def age(predictions: List[str], ground_truths: List[str], demographics: List[str], similarities: List[float] = None) -> Dict[str, float]:
    groups = {
        "a1": {"preds": [], "gts": [], "sims": []},
        "a2": {"preds": [], "gts": [], "sims": []},
        "a3": {"preds": [], "gts": [], "sims": []},
        "a4": {"preds": [], "gts": [], "sims": []},
    }

    # Compute similarities if not provided
    if similarities is None:
        similarities = compute_pairwise_similarities(predictions, ground_truths)

    for pred, gt, demo, sim in zip(predictions, ground_truths, demographics, similarities):
        if demo is not None and "age" in demo.lower():
            try:
                age_str = demo.split("age:")[1].strip().split()[0].replace(",", "")
                age_val = float(age_str)
            except (IndexError, ValueError):
                continue

            if age_val <= 25:
                groups["a1"]["preds"].append(pred)
                groups["a1"]["gts"].append(gt)
                groups["a1"]["sims"].append(sim)
            elif 25 < age_val <= 50:
                groups["a2"]["preds"].append(pred)
                groups["a2"]["gts"].append(gt)
                groups["a2"]["sims"].append(sim)
            elif 50 < age_val <= 75:
                groups["a3"]["preds"].append(pred)
                groups["a3"]["gts"].append(gt)
                groups["a3"]["sims"].append(sim)
            elif 75 < age_val:
                groups["a4"]["preds"].append(pred)
                groups["a4"]["gts"].append(gt)
                groups["a4"]["sims"].append(sim)

    results = {}
    acc_values = []
    f1_values = []
    sim_values = []
    tpr_values = []
    fpr_values = []
    fdr_values = []

    for group in ["a1", "a2", "a3", "a4"]:
        preds = groups[group]["preds"]
        gts = groups[group]["gts"]
        sims = groups[group]["sims"]
        if len(preds) == 0:
            continue
        metrics = compute_dataset_metrics(preds, gts, sims)["dataset_metrics"]
        acc = metrics["accuracy"]
        f1 = metrics["f1"]
        sim = metrics["similarity"]
        tpr = metrics.get("tpr", metrics["sensitivity"])
        fpr = metrics.get("fpr", 1.0 - metrics["specificity"])
        fdr = metrics.get("fdr", 1.0 - metrics["precision"])

        results[f"{group}/accuracy"] = acc
        results[f"{group}/f1"] = f1
        results[f"{group}/similarity"] = sim
        results[f"{group}/tpr"] = tpr
        results[f"{group}/fpr"] = fpr
        results[f"{group}/fdr"] = fdr

        acc_values.append(acc)
        f1_values.append(f1)
        sim_values.append(sim)
        tpr_values.append(tpr)
        fpr_values.append(fpr)
        fdr_values.append(fdr)

    results["acc"] = sum(acc_values) / len(acc_values)
    results["f1"] = sum(f1_values) / len(f1_values)
    results["similarity"] = sum(sim_values) / len(sim_values)
    results["tpr"] = sum(tpr_values) / len(tpr_values)
    results["fpr"] = sum(fpr_values) / len(fpr_values)
    results["fdr"] = sum(fdr_values) / len(fdr_values)

    if len(f1_values) >= 2:
        results["acc_diff"] = max(acc_values) - min(acc_values)
        results["acc_std"] = statistics.stdev(acc_values)
        print(f"Accuracy max diff = {results['acc_diff']:.4f}")
        print(f"std of accuracy for age = {results['acc_std']:.4f}")
    if len(f1_values) >= 2:
        results["f1_diff"] = max(f1_values) - min(f1_values)
        results["f1_std"] = statistics.stdev(f1_values)
        print(f"F1 max diff = {results['f1_diff']:.4f}")
        print(f"std of f1 for age = {results['f1_std']:.4f}")
    if len(sim_values) >= 2:
        results["similarity_diff"] = max(sim_values) - min(sim_values)
        results["similarity_std"] = statistics.stdev(sim_values)
        print(f"Similarity max diff = {results['similarity_diff']:.4f}")
        print(f"std of similarity for age = {results['similarity_std']:.4f}")
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


def compute_dataset_metrics(predictions: List[str], ground_truths: List[str], similarities: List[float] = None) -> Dict[str, Dict]:
    """
    Compute metrics for a single dataset, with class-wise averaging.

    Args:
        predictions (List[str]): List of model predictions for this dataset.
        ground_truths (List[str]): List of ground truth labels for this dataset.
        similarities (List[float]): Optional precomputed similarity scores for each pair.

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

    # Compute similarity score (average of precomputed pairwise similarities)
    if similarities is not None and len(similarities) > 0:
        dataset_metrics["similarity"] = sum(similarities) / len(similarities)
    else:
        # Compute similarities if not provided
        sims = compute_pairwise_similarities(predictions, ground_truths)
        dataset_metrics["similarity"] = sum(sims) / len(sims) if sims else 0.0

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
            os.path.join(output_dir, f"input_data_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"),
            "w"
    ) as f:
        json.dump(input_data, f, indent=4)

    # Group examples by data source and dataset
    grouped_data = defaultdict(lambda: defaultdict(lambda: {"preds": [], "gts": [], "demos": []}))

    for pred, gt, source, dataset, demo in zip(predictions, ground_truths, data_sources, datasets, demographics):
        grouped_data[source][dataset]["preds"].append(pred)
        grouped_data[source][dataset]["gts"].append(gt)
        grouped_data[source][dataset]["demos"].append(demo)

    # Initialize the flattened result dictionary
    result: Dict[str, float] = {}

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
    total_datasets = 0

    overall_acc = []
    overall_f1 = []
    overall_similarity = []
    overall_tpr = []
    overall_fpr = []
    overall_fdr = []
    overall_acc_diff = []
    overall_f1_diff = []
    overall_similarity_diff = []
    overall_tpr_diff = []
    overall_fpr_diff = []
    overall_fdr_diff = []
    overall_acc_std = []
    overall_f1_std = []
    overall_similarity_std = []

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
            dataset_predictions: List[str] = dataset_data["preds"]
            dataset_ground_truths: List[str] = dataset_data["gts"]
            dataset_demographics: List[str] = dataset_data["demos"]

            # Store dataset-level metrics with the format "data_source/dataset/metric"
            for metric_name, metric_value in dataset_result["dataset_metrics"].items():
                result[f"{source_name}/{dataset_name}/{metric_name}"] = metric_value

            # Skip empty datasets
            if dataset_result["active_classes"] == 0:
                continue

            total_datasets_in_source += 1
            total_datasets += 1

            # Accumulate metrics for data source average (equal dataset weighting)
            for metric_name in source_metrics.keys():
                source_metrics[metric_name] += dataset_result["dataset_metrics"][metric_name]

            # Accumulate for global metrics (equal dataset weighting)
            for metric_name in global_metrics.keys():
                global_metrics[metric_name] += dataset_result["dataset_metrics"][metric_name]

            acc_diffs = []
            f1_diffs = []
            similarity_diffs = []
            tpr_diffs = []
            fpr_diffs = []
            fdr_diffs = []
            acc_stds = []
            f1_stds = []
            similarity_stds = []
            accs, f1s, similarities, tprs, fprs, fdrs = [], [], [], [], [], []

            try:
                gender_results = gender(dataset_predictions, dataset_ground_truths, dataset_demographics)
                acc_diffs.append(gender_results["acc_diff"])
                f1_diffs.append(gender_results["f1_diff"])
                similarity_diffs.append(gender_results.get("similarity_diff", 0.0))
                tpr_diffs.append(gender_results["tpr_diff"])
                fpr_diffs.append(gender_results["fpr_diff"])
                fdr_diffs.append(gender_results["fdr_diff"])
                acc_stds.append(gender_results["acc_std"])
                f1_stds.append(gender_results["f1_std"])
                similarity_stds.append(gender_results.get("similarity_std", 0.0))
                accs.append(gender_results["acc"])
                f1s.append(gender_results["f1"])
                similarities.append(gender_results.get("similarity", 0.0))
                tprs.append(gender_results["tpr"])
                fprs.append(gender_results["fpr"])
                fdrs.append(gender_results["fdr"])
                for k, v in gender_results.items():
                    result[f"fairness/{dataset_name}/gender/{k}"] = v
            except ZeroDivisionError:
                pass

            try:
                age_results = age(dataset_predictions, dataset_ground_truths, dataset_demographics)
                acc_diffs.append(age_results["acc_diff"])
                f1_diffs.append(age_results["f1_diff"])
                similarity_diffs.append(age_results.get("similarity_diff", 0.0))
                tpr_diffs.append(age_results["tpr_diff"])
                fpr_diffs.append(age_results["fpr_diff"])
                fdr_diffs.append(age_results["fdr_diff"])
                acc_stds.append(age_results["acc_std"])
                f1_stds.append(age_results["f1_std"])
                similarity_stds.append(age_results.get("similarity_std", 0.0))
                accs.append(age_results["acc"])
                f1s.append(age_results["f1"])
                similarities.append(age_results.get("similarity", 0.0))
                tprs.append(age_results["tpr"])
                fprs.append(age_results["fpr"])
                fdrs.append(age_results["fdr"])
                for k, v in age_results.items():
                    result[f"fairness/{dataset_name}/age/{k}"] = v

                # parent_results = parent(predictions, ground_truths, demographics)
                # for k, v in parent_results.items():
                #     result[f"fairness/{dataset_name}/parent/{k}"] = v
                avg_acc = sum(accs) / len(accs)
                result[f"fairness/{dataset_name}/avg_acc"] = avg_acc
                overall_acc.append(avg_acc)
                avg_f1 = sum(f1s) / len(f1s)
                result[f"fairness/{dataset_name}/avg_f1"] = avg_f1
                overall_f1.append(avg_f1)
                avg_similarity = sum(similarities) / len(similarities)
                result[f"fairness/{dataset_name}/avg_similarity"] = avg_similarity
                overall_similarity.append(avg_similarity)
                avg_tpr = sum(tprs) / len(tprs)
                result[f"fairness/{dataset_name}/avg_tpr"] = avg_tpr
                overall_tpr.append(avg_tpr)
                avg_fpr = sum(fprs) / len(fprs)
                result[f"fairness/{dataset_name}/avg_fpr"] = avg_fpr
                overall_fpr.append(avg_fpr)
                avg_fdr = sum(fdrs) / len(fdrs)
                result[f"fairness/{dataset_name}/avg_fdr"] = avg_fdr
                overall_fdr.append(avg_fdr)

                avg = sum(acc_diffs) / len(acc_diffs)
                result[f"fairness/{dataset_name}/avg_acc_diff"] = avg
                print(f"[fairness/{dataset_name}] avg_acc_diff = {avg:.4f}")
                std = sum(acc_stds) / len(acc_stds)
                result[f"fairness/{dataset_name}/std_acc"] = std
                print(f"[fairness/{dataset_name}] std_acc = {std:.4f}")
                overall_acc_std.append(std)
                overall_acc_diff.append(avg)

                avg = sum(f1_diffs) / len(f1_diffs)
                result[f"fairness/{dataset_name}/avg_f1_diff"] = avg
                print(f"[fairness/{dataset_name}] avg_f1_diff = {avg:.4f}")
                std = sum(f1_stds) / len(f1_stds)
                result[f"fairness/{dataset_name}/f1_std"] = std
                print(f"[fairness/{dataset_name}] f1_std = {std:.4f}")
                overall_f1_std.append(std)
                overall_f1_diff.append(avg)

                avg = sum(similarity_diffs) / len(similarity_diffs) if similarity_diffs else 0.0
                result[f"fairness/{dataset_name}/avg_similarity_diff"] = avg
                print(f"[fairness/{dataset_name}] avg_similarity_diff = {avg:.4f}")
                std = sum(similarity_stds) / len(similarity_stds) if similarity_stds else 0.0
                result[f"fairness/{dataset_name}/similarity_std"] = std
                print(f"[fairness/{dataset_name}] similarity_std = {std:.4f}")
                overall_similarity_std.append(std)
                overall_similarity_diff.append(avg)

                avg = sum(tpr_diffs) / len(tpr_diffs)
                result[f"fairness/{dataset_name}/avg_tpr_diff"] = avg
                print(f"[fairness/{dataset_name}] avg_tpr_diff = {avg:.4f}")
                overall_tpr_diff.append(avg)

                avg = sum(fpr_diffs) / len(fpr_diffs)
                result[f"fairness/{dataset_name}/avg_fpr_diff"] = avg
                print(f"[fairness/{dataset_name}] avg_fpr_diff = {avg:.4f}")
                overall_fpr_diff.append(avg)

                avg = sum(fdr_diffs) / len(fdr_diffs)
                result[f"fairness/{dataset_name}/avg_fdr_diff"] = avg
                print(f"[fairness/{dataset_name}] avg_fdr_diff = {avg:.4f}")
                overall_fdr_diff.append(avg)
            except ZeroDivisionError:
                pass

        # Calculate data source average (equal dataset weighting)
        if total_datasets_in_source > 0:
            for metric_name in source_metrics.keys():
                source_metrics[metric_name] /= total_datasets_in_source

            # Store data source metrics with the format "data_source/metric"
            for metric_name, metric_value in source_metrics.items():
                result[f"{source_name}/{metric_name}"] = metric_value

            total_data_sources += 1

    # Calculate global average (equal data source weighting)
    if total_datasets > 0:
        for metric_name in global_metrics.keys():
            global_metrics[metric_name] /= total_datasets

        # Store global metrics with the format "val/metric"
        for metric_name, metric_value in global_metrics.items():
            result[f"val/{metric_name}"] = metric_value

    result["val/num_data_sources"] = total_data_sources
    result["val/num_datasets"] = total_datasets

    try:
        result[f"overall/overall_acc"] = sum(overall_acc) / len(overall_acc)
        result[f"overall/overall_f1"] = sum(overall_f1) / len(overall_f1)
        result[f"overall/overall_similarity"] = sum(overall_similarity) / len(overall_similarity) if overall_similarity else 0.0
        result[f"overall/overall_tpr"] = sum(overall_tpr) / len(overall_tpr)
        result[f"overall/overall_fpr"] = sum(overall_fpr) / len(overall_fpr)
        result[f"overall/overall_fdr"] = sum(overall_fdr) / len(overall_fdr)
        result[f"overall/overall_acc_diff"] = sum(overall_acc_diff) / len(overall_acc_diff)
        result[f"overall/overall_f1_diff"] = sum(overall_f1_diff) / len(overall_f1_diff)
        result[f"overall/overall_similarity_diff"] = sum(overall_similarity_diff) / len(overall_similarity_diff) if overall_similarity_diff else 0.0
        result[f"overall/overall_tpr_diff"] = sum(overall_tpr_diff) / len(overall_tpr_diff)
        result[f"overall/overall_fpr_diff"] = sum(overall_fpr_diff) / len(overall_fpr_diff)
        result[f"overall/overall_fdr_diff"] = sum(overall_fdr_diff) / len(overall_fdr_diff)
        result[f"overall/overall_acc_std"] = sum(overall_acc_std) / len(overall_acc_std)
        result[f"overall/overall_f1_std"] = sum(overall_f1_std) / len(overall_f1_std)
        result[f"overall/overall_similarity_std"] = sum(overall_similarity_std) / len(overall_similarity_std) if overall_similarity_std else 0.0
        result[f"overall/acc_es"] = result[f"overall/overall_acc"] / (1 + result[f"overall/overall_acc_std"])
        result[f"overall/f1_es"] = result[f"overall/overall_f1"] / (1 + result[f"overall/overall_f1_std"])
        result[f"overall/similarity_es"] = result[f"overall/overall_similarity"] / (1 + result[f"overall/overall_similarity_std"]) if result[f"overall/overall_similarity_std"] > 0 else result[f"overall/overall_similarity"]
        for key in [
            "overall/overall_tpr",
            "overall/overall_fpr",
            "overall/overall_fdr",
            "overall/overall_acc_diff",
            "overall/overall_f1_diff",
            "overall/overall_similarity_diff",
            "overall/overall_tpr_diff",
            "overall/overall_fpr_diff",
            "overall/overall_fdr_diff",
            "overall/overall_acc_std",
            "overall/overall_f1_std",
            "overall/overall_similarity_std",
        ]:
            print(f"{key}/{result[key]:.4f}")
    except KeyError:
        print("No fairness metrics computed.")
    except ZeroDivisionError:
        print("Division by zero, no fairness metrics computed.")

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