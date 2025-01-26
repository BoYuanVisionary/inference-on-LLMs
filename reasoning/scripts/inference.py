import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from reasoning.tools.utils import load_model_with_vllm
from reasoning.inference.greedy import GreedyInference
from datasets import load_dataset
import wandb
import torch.distributed as dist
import time
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="greedy")
    parser.add_argument("--batch_size", type=int, default=10)
    return parser.parse_args()

# only for greedy inference

args = get_args()
wandb_name = f"{args.method}_inference"
wandb.init(project="efficient_reasoning",name=wandb_name)
wandb.config.update({"model": "Llama-3.2-3B-Instruct", "batch_size": args.batch_size})


number_return_sequences = [1] 

dataset = load_dataset("HuggingFaceH4/MATH-500")
dataset = dataset['test']
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model, tokenizer = load_model_with_vllm(model_name,tensor_parallel_size=1)
tokenizer.pad_token = tokenizer.eos_token
inference = GreedyInference(model, tokenizer)

batch_size = 10
# make a batch of samples
wandb.log({"num_return_sequences": number_return_sequences})
start_time = time.time()
for i in range(0, len(dataset), batch_size):
    questions = dataset['problem'][i:i+batch_size]
    answers = dataset['solution'][i:i+batch_size]
    accuracy = inference.inference(questions, answers)
    print("Accuracy: {}".format(accuracy))
end_time = time.time()
print("Time taken: {} seconds".format(end_time - start_time))
inference.reset()
wandb.log({"accuracy": accuracy})
wandb.finish()
    
if dist.is_initialized():
    dist.destroy_process_group()