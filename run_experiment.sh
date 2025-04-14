#!/bin/bash

# Default values
CONFIG_PATH="configs/majority_llama.yaml"
MODEL="meta-llama/Llama-3.2-1B-Instruct"
NUM_SEQUENCES=16
INFERENCE_METHOD="majority"
BATCH_SIZE=10
TEMPERATURE=0.7
TOP_P=0.9

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c, --config         Config file path (default: configs/majority_llama.yaml)"
    echo "  -m, --model         Model name/path"
    echo "  -n, --num-sequences Number of return sequences"
    echo "  -i, --inference     Inference method (majority/sampling/etc)"
    echo "  -b, --batch-size    Batch size"
    echo "  -t, --temperature   Temperature for sampling"
    echo "  -p, --top-p        Top p for sampling"
    echo "  -h, --help         Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -n|--num-sequences)
            NUM_SEQUENCES="$2"
            shift 2
            ;;
        -i|--inference)
            INFERENCE_METHOD="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -p|--top-p)
            TOP_P="$2"
            shift 2
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

# Create a temporary config file with updated parameters
TMP_CONFIG=$(mktemp)
cat "$CONFIG_PATH" > "$TMP_CONFIG"

# Update the parameters in the temporary config
sed -i "s|model_name:.*|model_name: $MODEL|" "$TMP_CONFIG"
sed -i "s|num_return_sequences:.*|num_return_sequences: $NUM_SEQUENCES|" "$TMP_CONFIG"
sed -i "s|inference_method:.*|inference_method: $INFERENCE_METHOD|" "$TMP_CONFIG"
sed -i "s|batch_size:.*|batch_size: $BATCH_SIZE|" "$TMP_CONFIG"
sed -i "s|temperature:.*|temperature: $TEMPERATURE|" "$TMP_CONFIG"
sed -i "s|top_p:.*|top_p: $TOP_P|" "$TMP_CONFIG"

# Update config name to reflect changes
CONFIG_NAME="$(basename "$MODEL")-${INFERENCE_METHOD}_${NUM_SEQUENCES}"
sed -i "s|config_name:.*|config_name: $CONFIG_NAME|" "$TMP_CONFIG"

# Print the command that will be executed
echo "Running with parameters:"
echo "Config file: $TMP_CONFIG"
echo "Model: $MODEL"
echo "Num sequences: $NUM_SEQUENCES"
echo "Inference method: $INFERENCE_METHOD"
echo "Batch size: $BATCH_SIZE"
echo "Temperature: $TEMPERATURE"
echo "Top p: $TOP_P"

# Run the Python script with the temporary config
python main.py --config "$TMP_CONFIG"

# Clean up
rm "$TMP_CONFIG" 