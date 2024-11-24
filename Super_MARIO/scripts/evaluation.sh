CUDA_VISIBLE_DEVICES=3 python solver_demo.py \
--custom_cfg configs/sbs_sft.yaml \
--qaf MARIO_EVAL/data/math_testset_annotation.json

# python eval_output_jsonl.py \
# --res_file MARIO_EVAL/data/math_testset_annotation.json

python eval_output_jsonl.py --res_file MARIO_EVAL/data/math_testset_annotation.json.AlphaMath-7B.jsonl --react

CUDA_VISIBLE_DEVICES=1 python solver_demo.py \
--custom_cfg configs/mcts_round1.yaml \
--qaf  MARIO_EVAL/data/math_testset_annotation.json


# MCTS inference
CUDA_VISIBLE_DEVICES=1 python solver_demo.py \
--custom_cfg configs/mcts_sft.yaml \
--qaf MARIO_EVAL/data/math_testset_annotation.json