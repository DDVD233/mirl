#!/bin/bash

# LLM Judge Evaluation Script
# This script runs the LLM judge evaluation with configurable parameters


# Default values
# PLEASE PUT THE RESULTS_PATH as the path within the save dir that points to the generated outputs for all questions, which should look like "full_test_or_val_generation_outputs/150.json"
# RESULTS_PATH="/Users/keane/Desktop/research/human-behavior/verl/examples/reward_function/temp/input_data_2025-11-28_11-53-44.json"
# SAVE_PATH="/Users/keane/Desktop/research/human-behavior/verl/examples/reward_function/temp/results_temp.json"

RESULTS_PATH="/home/keaneong/human-behavior/verl/verl_independent_test_eval/results/google_gemma-4-e4b-it_all_judge.json"
SAVE_PATH="/home/keaneong/human-behavior/verl/verl_independent_test_eval/results/gemma4_llm_grading_results.json"
PROVIDER="openai"
WANDB_PROJECT="llm_judge_eval"
WANDB_RUN_NAME="gemma4_experiment_$(date +%Y%m%d_%H%M%S)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --predictions_to_eval_path)
            RESULTS_PATH="$2"
            shift 2
            ;;
        --grading_results_path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_run_name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --no-wandb)
            WANDB_PROJECT=""
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --predictions_to_eval_path PATH       Path to the results JSON file"
            echo "  --grading_results_path PATH          Path to save graded results as JSON"
            echo "  --provider PROVIDER       LLM provider (openai or anthropic)"
            echo "  --wandb_project PROJECT   W&B project name"
            echo "  --wandb_run_name NAME     W&B run/experiment name"
            echo "  --no-wandb                Disable W&B logging"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --predictions_to_eval_path input.json --wandb_run_name my_experiment"
            echo "  $0 --provider anthropic --no-wandb"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if required environment variables are set
if [ "$PROVIDER" = "openai" ] && [ -z "$MIT_OPENAI_API_KEY" ]; then
    echo "Error: MIT_OPENAI_API_KEY environment variable is not set"
    echo "Please export your OpenAI API key:"
    echo "  export MIT_OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

if [ "$PROVIDER" = "anthropic" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY environment variable is not set"
    echo "Please export your Anthropic API key:"
    echo "  export ANTHROPIC_API_KEY='your-api-key-here'"
    exit 1
fi

# Check if results file exists
if [ ! -f "$RESULTS_PATH" ]; then
    echo "Error: Results file not found: $RESULTS_PATH"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "LLM Judge Evaluation Configuration"
echo "=========================================="
echo "Results path:    $RESULTS_PATH"
echo "Save path:       $SAVE_PATH"
echo "Provider:        $PROVIDER"
if [ -n "$WANDB_PROJECT" ]; then
    echo "W&B project:     $WANDB_PROJECT"
    echo "W&B run name:    $WANDB_RUN_NAME"
else
    echo "W&B:             Disabled"
fi
echo "=========================================="
echo ""

# Build the command
CMD="python llm_judge_eval.py --predictions_to_eval_path \"$RESULTS_PATH\" --grading_results_path \"$SAVE_PATH\" --provider $PROVIDER"

if [ -n "$WANDB_PROJECT" ]; then
    CMD="$CMD --wandb_project \"$WANDB_PROJECT\" --wandb_run_name \"$WANDB_RUN_NAME\""
fi

# Run the evaluation
echo "Starting evaluation..."
echo ""
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed with exit code $?"
    echo "=========================================="
    exit 1
fi
