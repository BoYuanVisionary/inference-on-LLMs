from reasoning.tools.utils import load_model_with_vllm
from reasoning.evaluator.math_grader import math_equal, extract_answer
from datasets import load_dataset
from vllm import SamplingParams
import wandb
import time
from reasoning.tools.logger import load_config, apply_config
import numpy as np
import argparse
from reasoning.inference.tree import Path
from reasoning.inference.base import BaseInference
from reasoning.models.model import ValueModel_qwen
# Supported inference methods: majority_voting, greedy (majority_voting with 1 step), weighted_majority_voting, best_of_N

class MajorityInference(BaseInference):
    
    def __init__(self, policy_model, tokenizer, sampling_params, config_name, reward_model, method):
        super().__init__(policy_model, tokenizer, sampling_params, config_name, reward_model, method)
        
        if method not in self.method_dict.keys() or method == 'beam_search':
            raise ValueError("Inference method not supported yet.")
        

    def inference(self, system_prompt, questions, answers):

        inference_method = self.method_dict[self.method]
        self.sample_size += len(questions)
        num_return_sequences = self.sampling_params.n

        generated_solutions, num_generated_tokens, num_input_tokens = self.generate_text(system_prompt, questions)
        self.num_generated_tokens.extend(num_generated_tokens)
        self.num_input_tokens.extend(num_input_tokens)

        # Get rewards for each solution. The fault reward for PRM is the minimum reward for each step
        rewards = []
        if self.method in ['weighted_majority', 'best_of_N']:
            for i in range(len(questions)):
                current_generated_solutions = generated_solutions[i*num_return_sequences:(i+1)*num_return_sequences]
                for solution in current_generated_solutions:
                    reward_value = min(self.reward_model.get_value_with_steps(questions[i], Path(solution).steps))
                    # If the solution is not a valid solution, set the reward to 0. 
                    # To be more precise, run ablation to see if this is necessary
                    if extract_answer(solution) is None: 
                        reward_value = 0
                    rewards.append(reward_value)
        else: # no need to run reward model for majority voting
            rewards = None

        for i in range(len(questions)):
            current_generated_solutions = generated_solutions[i*num_return_sequences:(i+1)*num_return_sequences]
            current_rewards = rewards[i*num_return_sequences:(i+1)*num_return_sequences] if rewards is not None else None
            selected_solution = inference_method(current_generated_solutions, current_rewards)
            extracted_answer = extract_answer(answers[i])
            is_correct = math_equal(selected_solution, extracted_answer)
            if extracted_answer is None:
                raise ValueError('extracted answer is None')
            if is_correct:
                self.right_count += 1
            sum_num_generated_tokens = np.sum(num_generated_tokens[i*num_return_sequences:(i+1)*num_return_sequences])
            print(f'question: {questions[i]}')
            print(f'solution: {selected_solution}')
            print(f'extracted answer: {extracted_answer}')
            print(f'is correct: {is_correct}')
            print(f'sum_num_generated_tokens: {sum_num_generated_tokens}')
            print(f'current rewards: {current_rewards}')
            print("--------------------------------")

            save_data = {
                "question": questions[i],
                "generated_solutions": current_generated_solutions,
                "selected_solution": selected_solution,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "rewards": current_rewards,
            }
            self.save_solutions_to_jsonl(save_data)
        print(f'processed {self.sample_size} samples')

        return self.right_count / self.sample_size

    def majority_vote(self, generated_text, rewards=None):
        # extract the generated text for a single problem
        solutions = [extract_answer(text) for text in generated_text]
        print(f'found solutions: {solutions}')
        if len(solutions) == 1:
            return solutions[0]
        # create a matrix to store if any two solutions are equal
        equal_matrix = [[math_equal(solutions[i], solutions[j]) for j in range(i,len(solutions))] for i in range(len(solutions))]
        # majority vote: choose the solution whose number of equal solutions is the largest  (sum of each row)
        # if there are multiple solutions with the same number of equal solutions, choose the first one
        # print(equal_matrix)
        majority_solution = solutions[max(range(len(equal_matrix)), key=lambda i: sum(equal_matrix[i]))]
        return majority_solution
    
    def weighted_majority_vote(self, generated_text, rewards):
        solutions = [extract_answer(text) for text in generated_text]
        print(f'found solutions: {solutions}')
        print(f'rewards: {rewards}')
        if len(solutions) == 1:
            return solutions[0]
            
        # Create a weighted vote counter for each solution
        weighted_votes = [0.0] * len(solutions)
        
        # For each solution, add its reward to all solutions that are equal to it
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                if math_equal(solutions[i], solutions[j]):
                    weighted_votes[i] += rewards[j]
        
        # print(f'weighted votes: {weighted_votes}')
        
        # Choose the solution with the highest weighted vote
        majority_solution = solutions[max(range(len(weighted_votes)), key=lambda i: weighted_votes[i])]
        return majority_solution

    def best_of_N(self, generated_text, rewards):
        solutions = [extract_answer(text) for text in generated_text]
        print(f'found solutions: {solutions}')
        # return the solution with the highest reward, if there are multiple solutions with the same reward, choose the first one
        best_solution = solutions[max(range(len(rewards)), key=lambda i: rewards[i])]
        return best_solution

if __name__ == "__main__":
    
    # use parser to parse the config file
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
    # need to set gpu_memory_utilization to 0.8 to avoid OOM when using qwen2.5b with 16
    model, tokenizer = load_model_with_vllm(model_name, task='auto', tensor_parallel_size=len(config["cuda_device_ids"]), gpu_memory_utilization=0.8)
    tokenizer.pad_token = tokenizer.eos_token 
    reward_model = config.get("reward_model", None)
    reward_model = ValueModel_qwen(device = "auto")

    # inference hyperparameters
    num_return_sequences = config["num_return_sequences"]
    max_new_tokens = config["max_new_tokens"]
    temperature = config["temperature"]
    top_p = config["top_p"]
    batch_size = config["batch_size"]
    num_return_sequences = config["num_return_sequences"]
    system_prompt = config["system_prompt"]

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

    inference = MajorityInference(model, tokenizer,sampling_params,config_name,reward_model,method=config["inference_method"])
    # dataset = dataset.shuffle(seed=42).select(range(100))
    start_time = time.time()
    for i in range(0, len(dataset), batch_size):
        questions = dataset['problem'][i:i+batch_size]
        answers = dataset['solution'][i:i+batch_size]            
        accuracy = inference.inference(system_prompt, questions, answers)
        print("Accuracy: {}".format(accuracy))
        # inference.reset()
    wandb.log({"accuracy": accuracy})
    end_time = time.time()
    print(f"parallel size: {num_return_sequences}")
    print(f"generated tokens per sample in average: {np.mean(inference.num_generated_tokens) * num_return_sequences}")
    wandb.log({"generated tokens per sample in average": np.mean(inference.num_generated_tokens) * num_return_sequences})
    print("Time taken: {} seconds".format(end_time - start_time))

    wandb.finish()


        
    

