#!/bin/bash

# Default values
MAX_PARALLEL=4  # Number of parallel processes
CONFIG_DIR="configs/ablation_study"  # Directory containing YAML configs
LOG_DIR="logs/ablation_study"        # Directory for logs
SLEEP_TIME=60                        # Sleep time between launches (in seconds)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --sleep)
            SLEEP_TIME="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Function to count running jobs
count_running_jobs() {
    jobs -p | wc -l
}

# Function to run a single experiment
run_experiment() {
    config_file="$1"
    log_file="$2"
    
    # Extract experiment name from config file
    experiment_name=$(basename "$config_file" .yaml)
    
    echo "Starting experiment: $experiment_name"
    echo "Log file: $log_file"
    
    # Run the experiment and log output
    {
        echo "=== Experiment started at $(date) ==="
        echo "Config file: $config_file"
        if [[ "$config_file" == *"samplingTree"* ]]; then
            python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/samplingTree.py \
                --config "$config_file"
        elif [[ "$config_file" == *"majority"* ]]; then
            python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/majority.py \
                --config "$config_file"
        fi
        echo "=== Experiment finished at $(date) ==="
    } > "$log_file" 2>&1
}

# Find all yaml files
yaml_files=$(find "$CONFIG_DIR" -type f \( -name "*.yaml" -o -name "*.yml" \))

# Process each yaml file
for config_file in $yaml_files; do
    # Wait if we have too many running jobs
    while [ $(count_running_jobs) -ge $MAX_PARALLEL ]; do
        sleep 5
    done
    
    # Create log file name
    log_file="$LOG_DIR/$(basename "$config_file" .yaml).log"
    
    # Run experiment in background
    run_experiment "$config_file" "$log_file" &
    
    # Sleep between launches to prevent overwhelming the system
    sleep $SLEEP_TIME
done

# Wait for all background jobs to complete
wait

echo "All experiments completed!"

# Summarize results
echo "=== Experiment Summary ===" > "$LOG_DIR/summary.txt"
for log_file in "$LOG_DIR"/*.log; do
    echo "=== $(basename "$log_file") ===" >> "$LOG_DIR/summary.txt"
    # Add your custom log parsing logic here
    # For example, extract final metrics, completion status, etc.
    grep "finished at" "$log_file" >> "$LOG_DIR/summary.txt"
done

echo "Summary written to $LOG_DIR/summary.txt" 