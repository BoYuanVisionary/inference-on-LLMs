python test_mcts.py --policy_model_device cuda:0 --reward_model_device cuda:1 --branch 2 --roll_branch 1 --roll_forward_steps 2 & # first 100 examples
python test_mcts.py --policy_model_device cuda:2 --reward_model_device cuda:3 --branch 2 --roll_branch 1 --roll_forward_steps 2 & # last 100 examples
wait

python test_mcts.py --policy_model_device cuda:0 --reward_model_device cuda:1 --branch 1 --roll_branch 1 --roll_forward_steps 2 # first 100 examples
python test_mcts.py --policy_model_device cuda:2 --reward_model_device cuda:3 --branch 1 --roll_branch 1 --roll_forward_steps 2 # last 100 examples
wait


python test_mcts.py --policy_model_device cuda:0 --reward_model_device cuda:1 --branch 3 --roll_branch 1 --roll_forward_steps 2 # first 100 examples
python test_mcts.py --policy_model_device cuda:2 --reward_model_device cuda:3 --branch 3 --roll_branch 1 --roll_forward_steps 2 # last 100 examples
wait

python test_mcts.py --policy_model_device cuda:0 --reward_model_device cuda:1 --branch 2 --roll_branch 2 --roll_forward_steps 2 # first 100 examples
python test_mcts.py --policy_model_device cuda:2 --reward_model_device cuda:3 --branch 2 --roll_branch 2 --roll_forward_steps 2 # last 100 examples
wait

python test_mcts.py --policy_model_device cuda:0 --reward_model_device cuda:1 --branch 2 --roll_branch 3 --roll_forward_steps 2 # first 100 examples
python test_mcts.py --policy_model_device cuda:2 --reward_model_device cuda:3 --branch 2 --roll_branch 3 --roll_forward_steps 2 # last 100 examples
wait