#!/bin/bash

# Find all yaml/yml files in the current directory and run appropriate script for each
find . -type f \( -name "*.yaml" -o -name "*.yml" \) | while read -r config_file; do
    if [[ "$config_file" == *"samplingTree"* ]]; then
        echo "Running samplingTree with config: $config_file"
        python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/samplingTree.py \
         --config "$config_file"
        sleep 60
    elif [[ "$config_file" == *"majority"* ]]; then
        echo "Running majority with config: $config_file"
        python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/majority.py \
         --config "$config_file"
        sleep 60
    else
        echo "Skipping $config_file - no matching script found"
    fi
done