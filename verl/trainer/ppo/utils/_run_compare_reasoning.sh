#!/bin/bash

# On engaging cluster:
STEP_FILE=""
JSONL_FILE=""
OUTPUT_FILE=""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the Python script
python3 "${SCRIPT_DIR}/compare_reasoning.py" "$STEP_FILE" "$JSONL_FILE" "$OUTPUT_FILE"
