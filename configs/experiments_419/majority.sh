#!/bin/bash

# List of seeds
SEEDS=(123)

# List of model names
MODEL_NAMES=(
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

# Config file
CONFIG_FILE="/ssdscratch/byuan48/efficient_reasoning/configs/experiments_419/majority_base.yaml"

# Loop through each combination of seed and model name
for SEED in "${SEEDS[@]}"; do
  for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Running with seed: $SEED and model: $MODEL_NAME"
    
    # Run the Python script with the specified parameters
    python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/majority.py \
      --config "$CONFIG_FILE" \
      --seed "$SEED" \
      --model_name "$MODEL_NAME"
    
    # Optional: add a small delay between runs
    sleep 60
  done
done

echo "All experiments completed!"