#!/bin/sh

# Find all yaml/yml files in the current directory and run appropriate script for each
find . -type f \( -name "*.yaml" -o -name "*.yml" \) | while read -r config_file; do
    case "$config_file" in
        *samplingTree*)
            echo "Running samplingTree with config: $config_file"
            python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/samplingTree.py \
             --config "$config_file"
            ;;
        *majority*)
            echo "Running majority with config: $config_file"
            python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/majority.py \
             --config "$config_file"
            ;;
        *)
            echo "Skipping $config_file - no matching script found"
            ;;
    esac
done 