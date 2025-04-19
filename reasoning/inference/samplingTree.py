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
from reasoning.inference.tree import Tree
from reasoning.inference.majority import MajorityInference
from reasoning.evaluator.math_grader import math_equal, extract_answer

from reasoning.models.model import ValueModel_qwen


# Does not support batch inference

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../../configs/development.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    apply_config(config) 
    print(config)

    # load dataset and model
    dataset = load_dataset(config["data_path"])
    dataset = dataset['test']
    model_name = config["model_name"]

    policy_model, tokenizer = load_model_with_vllm(model_name, task='auto', tensor_parallel_size=len(config["cuda_device_ids"]), gpu_memory_utilization=0.8)
    tokenizer.pad_token = tokenizer.eos_token 
    reward_model = ValueModel_qwen(device = "auto") # default one
    start_problem_index = config.get("begin_problem_index", 0)
    end_problem_index = config.get("end_problem_index", len(dataset)-1)
    dataset = dataset.select(range(start_problem_index, end_problem_index))

    # inference hyperparameters
    num_return_sequences = config["num_return_sequences"]
    assert num_return_sequences == 1
    max_new_tokens = config["max_new_tokens"]
    temperature = config["temperature"]
    top_p = config["top_p"]
    system_prompt = config["system_prompt"]

    # SamplingTree parameters
    sampling_method = config["sampling_method"]
    samplingTree_temperature = config["samplingTree_temperature"]
    beam_width = config["beam_width"]
    max_steps = config["max_steps"]
    threshold = config["threshold"]
    ORM_type = config["ORM_type"]
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
        seed  = config["seed"]
    ) # shouldn't set seed for random sampling

    start_time = time.time()
    right_count = 0
    num_generated_tokens = 0
    error_steps_count_all = 0
    print(f'length of the dataset: {len(dataset)}')
    for i in range(0, len(dataset)):
        question = dataset['problem'][i]
        answer = dataset['solution'][i]        

        tree = Tree(system_prompt, question, policy_model, reward_model, sampling_method, config_name, sampling_params, samplingTree_temperature, beam_width, threshold, ORM_type)
        for _ in range(max_steps):
            path = tree.generate_next_trajectory() 
            tree.paths.append(path)
        majority_inference = MajorityInference(policy_model=policy_model, tokenizer=tokenizer, sampling_params=sampling_params, config_name=config_name, reward_model=reward_model, method = 'weighted_majority', ORM_type=ORM_type)
        all_solutions  = [path.solutions for path in tree.explored_paths]
        rewards = [tree.get_reward(path.scores) for path in tree.explored_paths]
        
        solution = majority_inference.weighted_majority_vote(all_solutions, rewards)   
        extracted_answer = extract_answer(answer)
        is_correct = math_equal(extracted_answer, solution)
        if extracted_answer is None:
            raise ValueError('extracted answer is None')
        if is_correct:
            right_count += 1
        num_generated_tokens += tree.num_generated_tokens
        print("--------------------------------")
        print(f'question: {question}')
        print(f'solution: {solution}')
        print(f'extracted answer: {extracted_answer}')
        print(f'is correct: {is_correct}')
        print(f'Accuracy: {right_count/(i+1)}')
        wandb.log({"Accuracy": right_count/(i+1)}, step=i)
        print(f'Number of generated tokens: {tree.num_generated_tokens}')
        # print out the error steps count
        error_steps_count = 0
        for path in tree.explored_paths:
            if len(path.steps) <= 1:
                error_steps_count += 1
        error_steps_count_all += error_steps_count
        print(f'Error steps count: {error_steps_count_all}')
        wandb.log({"Error steps count": error_steps_count_all}, step=i)
        print("--------------------------------")


    wandb.log({"Accuracy": right_count/len(dataset)})
    wandb.log({"Number of generated tokens": num_generated_tokens/len(dataset)})
    end_time = time.time()
    print("Time taken: {} seconds".format(end_time - start_time))

    wandb.finish()