python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/samplingTree.py \
    --config /ssdscratch/byuan48/efficient_reasoning/configs/samplingTree_llama_16_version5.yaml
sleep 60

python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/samplingTree.py \
    --config /ssdscratch/byuan48/efficient_reasoning/configs/samplingTree_qwen_16_version5.yaml
sleep 60

python /ssdscratch/byuan48/efficient_reasoning/reasoning/inference/samplingTree.py \
    --config /ssdscratch/byuan48/efficient_reasoning/configs/samplingTree_llama_16_version4.yaml
sleep 60

echo "Done"