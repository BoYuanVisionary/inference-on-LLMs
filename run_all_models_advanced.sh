#!/bin/bash

# Default values
CONFIG_DIR="/ssdscratch/byuan48/efficient_reasoning/configs"
WAIT_TIME=30
DRY_RUN=false
MODELS=("llama" "qwen")
PYTHON_SCRIPT="majority.py"

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -d, --dir           Config directory path (default: /ssdscratch/byuan48/efficient_reasoning/configs)"
    echo "  -w, --wait          Wait time between experiments in seconds (default: 30)"
    echo "  -m, --models        Models to run, comma-separated (default: llama,qwen)"
    echo "  -s, --script        Python script to run (default: majority.py)"
    echo "  --dry-run          Print commands without executing"
    echo "  -h, --help         Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        -w|--wait)
            WAIT_TIME="$2"
            shift 2
            ;;
        -m|--models)
            IFS=',' read -r -a MODELS <<< "$2"
            shift 2
            ;;
        -s|--script)
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Function to run an experiment
run_experiment() {
    local config_file="$1"
    echo "=================================================="
    echo "Running experiment with config: $config_file"
    echo "=================================================="
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run: python $PYTHON_SCRIPT --config $config_file"
    else
        python "$PYTHON_SCRIPT" --config "$config_file"
        
        # Wait between runs
        echo "Waiting $WAIT_TIME seconds before next experiment..."
        sleep "$WAIT_TIME"
    fi
}

# Function to find and run configs for a specific model
run_model_configs() {
    local model="$1"
    echo "Running ${model^} configurations..."  # ${model^} capitalizes first letter
    
    # Handle both lowercase and uppercase model names in config files
    local model_lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
    local model_upper=$(echo "$model" | tr '[:lower:]' '[:upper:]')
    local model_capital=${model_lower^}
    
    # Find matching config files
    for config in "$CONFIG_DIR"/*"$model_lower"*.yaml "$CONFIG_DIR"/*"$model_upper"*.yaml "$CONFIG_DIR"/*"$model_capital"*.yaml; do
        if [ -f "$config" ]; then
            run_experiment "$config"
        fi
    done
}

# Main execution
echo "Starting experiments with the following settings:"
echo "Config directory: $CONFIG_DIR"
echo "Wait time: $WAIT_TIME seconds"
echo "Models: ${MODELS[*]}"
echo "Python script: $PYTHON_SCRIPT"
echo "Dry run: $DRY_RUN"
echo ""

# Run configs for each specified model
for model in "${MODELS[@]}"; do
    run_model_configs "$model"
done

echo "All experiments completed!" 