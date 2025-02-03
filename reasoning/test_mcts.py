
from reasoning.models.model import Model
from reasoning.MCTS.task import MCTS_Task
from reasoning.tools.utils import seed_everything
seed_everything(110)
from reasoning.models.model import ValueModel_shepherd, ValueModel_qwen
import argparse
import datasets
import wandb
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_model_device', type=str, default='cuda:0')
    parser.add_argument('--reward_model_device', type=str, default='cuda:1')
    parser.add_argument('--seed', type=int, default=110)
    parser.add_argument('--iteration_limit', type=int, default=50)
    parser.add_argument('--end_gate', type=float, default=0.9)
    parser.add_argument('--branch', type=int, default=2)
    parser.add_argument('--roll_branch', type=int, default=1)
    parser.add_argument('--roll_forward_steps', type=int, default=2)
        
    return parser.parse_args()

args = get_args()


wandb.init(project='efficient_reasoning', name='mcts',config=args.__dict__)



seed_everything(args.seed)
policy_model = Model('meta-llama/Llama-3.2-3B-Instruct',device=args.policy_model_device)
reward_model = ValueModel_qwen(args.reward_model_device) # using qwen as reward model

# load math500 dataset
dataset = datasets.load_dataset("HuggingFaceH4/MATH-500")
dataset = dataset['test']
total_time = 0

# if policy_model_device is cuda:0, then use the first 100 examples otherwise use the last 100 examples
if args.policy_model_device in ['cuda:0', 'cuda:2']:
    problems = dataset[:100]['problem']
    solutions = dataset[:100]['solution']
else:
    problems = dataset[-100:]['problem']
    solutions = dataset[-100:]['solution']

number_right_answers = 0
for i in range(100):
    question = problems[i]
    answer = solutions[i]

    task = MCTS_Task(question, answer = answer, propose_method=policy_model, value_method=reward_model,
                     iteration_limit=args.iteration_limit, end_gate=args.end_gate, branch=args.branch,
                     roll_branch=args.roll_branch, roll_forward_steps=args.roll_forward_steps)
    time_start = time.time()
    output = task.run()
    time_end = time.time()
    total_time += time_end - time_start
    print(output)
    wandb.log({'question': question, 'answer': answer, 'output': output, 'time': time_end - time_start})
    number_right_answers += 1 if output['correctness'] else 0


print(f'total time: {total_time}')
print(f'average time: {total_time / len(dataset)}')
print(f'number of right answers: {number_right_answers}')
wandb.log({'total_time': total_time, 'average_time': total_time / len(dataset), 'number_right_answers': number_right_answers})

wandb.finish()


