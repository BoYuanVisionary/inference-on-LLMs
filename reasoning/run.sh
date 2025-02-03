python test_mcts.py --policy_model_device cuda:0 --reward_model_device cuda:0 --branch 2 --roll_branch 1 --roll_forward_steps 3 & # first 100 examples
python test_mcts.py --policy_model_device cuda:1 --reward_model_device cuda:1 --branch 2 --roll_branch 1 --roll_forward_steps 3 & # last 100 examples


# to do list check whether using the last step reward is better than using the mean reward of all steps

