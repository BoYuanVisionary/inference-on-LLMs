import argparse
import time
import numpy as np
import wandb

from reasoning.tools.utils import load_model_with_vllm
from datasets import load_dataset
from vllm import SamplingParams
from reasoning.inference.majority import MajorityInference
from reasoning.tools.logger import load_config, apply_config
import numpy as np
import argparse
from reasoning.samplingTree.tree import Tree
from reasoning.inference.majority import MajorityInference
from reasoning.evaluator.math_grader import math_equal, extract_answer

from reasoning.models.model_vllm import QwenPolicy
from reasoning.models.model import ValueModel_qwen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../../configs/development.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    apply_config(config) # set up wandb, seed and cuda device
    print(config)

    # load dataset and model
    dataset = load_dataset(config["data_path"])
    dataset = dataset['test']
    model_name = config["model_name"]

    value_model = ValueModel_qwen(device = "auto")
    policy_model = QwenPolicy(model_name, gpu_memory_utilization=0.8, max_new_tokens=2048)
    tokenizer = policy_model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token    

    # inference hyperparameters
    num_return_sequences = config["num_return_sequences"]
    max_new_tokens = config["max_new_tokens"]
    temperature = config["temperature"]
    top_p = config["top_p"]
    batch_size = config["batch_size"]
    num_return_sequences = config["num_return_sequences"]
    system_prompt = config["system_prompt"]

    # SamplingTree parameters
    sampling_method = config["sampling_method"]
    samplingTree_temperature = config["samplingTree_temperature"]
    threshold = config["threshold"]

    # name of the results file
    config_name = config["config_name"]

    # Also set the seed for the sampling params 
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=num_return_sequences,
        top_p=top_p,
        stop_token_ids=[tokenizer.eos_token_id],
        skip_special_tokens = True,
        include_stop_str_in_output = False,
    ) # shouldn't set seed for random sampling

    start_time = time.time()
    right_count = 0
    for i in range(0, 50):
        question = dataset['problem'][i]
        answer = dataset['solution'][i]        

        tree = Tree(system_prompt, question, policy_model, value_model, sampling_method='node')
        tree.set_proposal_params_node(temperature=samplingTree_temperature)
        for i in range(num_return_sequences):
            path = tree.generate_next_trajectory() 
        majority_inference = MajorityInference(model=None, tokenizer=None, sampling_params=None, config_name=None, method = 'majority')
        all_solutions  = [path.solutions for path in tree.paths]
        majority_solution = majority_inference.majority_vote(all_solutions)   
        extracted_answer = extract_answer(answer)
        is_correct = math_equal(extracted_answer, majority_solution)
        if extracted_answer is None:
            raise ValueError('extracted answer is None')
        if is_correct:
            right_count += 1
        print("--------------------------------")
        print(f'question: {question}')
        print(f'majority solution: {majority_solution}')
        print(f'extracted answer: {extracted_answer}')
        print(f'is correct: {is_correct}')
        print("--------------------------------")

    
    wandb.log({"accuracy": right_count/len(dataset)})
    end_time = time.time()
    print("Time taken: {} seconds".format(end_time - start_time))

    wandb.finish()