#!/bin/bash

python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/samplingTree.py \
    --config samplingTree_llama_128_09.yaml

sleep 60

python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/majority.py \
    --config Wmajority_llama_64.yaml

echo "Done"