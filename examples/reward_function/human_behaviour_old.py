import re
import json
from typing import Dict, List

import numpy
import torch
import numpy as np
from mathruler.grader import extract_boxed_content
import wandb
import random


def extract_boxed_content(text: str) -> str:
    """
    Extract content within \boxed{} or similar boxing notations.

    Args:
        text (str): Text containing potentially boxed content.

    Returns:
        str: Extracted boxed content or the original text if no box found.
    """

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

# def parse_conditions(text):
#     # Remove any boxing notation if present
#     text = text.replace("\\boxed{", "").replace("}", "")

#     # Split by common separators
#     for sep in [", ", " and ", " & ", ",", "&"]:
#         if sep in text:
#             return set(cond.strip() for cond in text.split(sep))

#     # If no separator found, treat as single condition
#     return {text.strip()}


# def parse_json(json_output):
#     """
#     Parsing out the markdown fencing from JSON code blocks.
#     """
#     # Look for content between ```json and ```
#     lines = json_output.splitlines()
#     for i, line in enumerate(lines):
#         if line == "```json" or line.strip() == "```":
#             json_output = "\n".join(lines[i + 1:])  # Remove everything before ```json
#             if "```" in json_output:
#                 json_output = json_output.split("```")[0]  # Remove everything after the closing ```
#             break  # Exit the loop once code block marker is found
#     return json_output


def extract_json_from_response(text):
    """
    Extract JSON content from markdown code blocks in the response.

    Args:
        text: The model's response text

    Returns:
        Parsed JSON object or None if no valid JSON found
    """
    # Find content between ```json and ```
    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_pattern, text)

    if not matches:
        return None

    # Try to parse each match as JSON
    for match in matches:
        try:
            parsed_json = json.loads(match.strip())
            return parsed_json
        except json.JSONDecodeError:
            continue

    # If we couldn't parse any match as valid JSON, try with ast.literal_eval
    import ast
    for match in matches:
        try:
            # Clean up the match a bit
            cleaned = match.strip().replace("'", "\"")
            parsed_json = ast.literal_eval(cleaned)
            return parsed_json
        except:
            continue

    return None

def format_reward(response: str) -> float:
    """
    Check whether the response matches the expected format.
    Here we require something like <think>...</think> ... \boxed{...}
    """
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def human_behaviour_compute_score_batch_f1(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[str], **kwargs) -> List[Dict[str, float]]:
    """
    Compute medical scoring for batch inputs including standard score, bounding box IoU, and format score.

    Args:
        data_sources: List of data sources (e.g., file paths or identifiers)
        solution_strs: List of model prediction strings
        ground_truths: List of ground truth strings
        extra_infos: List of extra information (e.g., segmentation masks, bounding boxes)

    Returns:
        List of score dictionaries
    """
    batch_scores = []

    # NOTE: At the per sample level, it just collapses to 1.0 and 0.0 for each component (for f1)

    for data_source, predict_str, ground_truth_label, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
      
        # Calculate standard score
        # Calculate format score (how well the JSON follows the expected format)
        # format_score = evaluate_bbox_format(predict_str)
        full_response = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)
        format_score = format_reward(full_response)
        pred_label = extract_boxed_content(predict_str).lower()
        ground_truth_label = ground_truth_label.lower()
     

        if pred_label == "None":
            standard_score = 0.0  # no answer
        else:
            # Parse both prediction and ground truth into sets of conditions
            # NOTE: considering that pred_label and ground truth are strings
            # one label each, how is f1 calculated?
            predicted_conditions = parse_conditions(pred_label)
            ground_truth_conditions = parse_conditions(ground_truth_label)

            # Calculate true positives, false positives, and false negatives
            true_positives = len(predicted_conditions.intersection(ground_truth_conditions))
            false_positives = len(predicted_conditions - ground_truth_conditions)
            false_negatives = len(ground_truth_conditions - predicted_conditions)

            # Calculate F1 score components
            precision = (
                true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            )
            recall = true_positives / (true_positives + false_negatives) if (
                                                                                        true_positives + false_negatives) > 0 else 0

            # Calculate F1 score (harmonic mean of precision and recall)
            standard_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # length score (ignore for now)
        # if len(predict_str) > 600:  # ~200 words
        #     length_score = 1
        # else:
        #     length_score = len(predict_str) * 0.001

        scores = {
            "score": 0.5 * standard_score + 0.3 * iou_score + 0.1 * format_score,
            "standard_score": standard_score,
            "format_score": format_score,
            # "length_score": length_score,
        }
        batch_scores.append(scores)

    return batch_scores