#!/bin/bash

# Base directory for configs
CONFIG_DIR="/ssdscratch/byuan48/efficient_reasoning/configs"

echo "Running Qwen configurations with 16..."

# Run all qwen configs with 16
for config in "$CONFIG_DIR"/*[qQ][wW][eE][nN]*16*.yaml; do
    if [ -f "$config" ]; then
        echo "=================================================="
        echo "Running: $config"
        echo "=================================================="
        python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/majority.py --config "$config"
        echo "Waiting 60 seconds..."
        sleep 60
    fi
done

echo "All experiments completed!"