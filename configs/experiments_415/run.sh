#!/bin/bash

# Find all yaml/yml files in the current directory and run samplingTree for each
find . -type f \( -name "*.yaml" -o -name "*.yml" \) | while read -r config_file; do
    echo "Running majotyi with config: $config_file"
    python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/samplingTree.py \
     --config "$config_file"
     sleep 60
done 