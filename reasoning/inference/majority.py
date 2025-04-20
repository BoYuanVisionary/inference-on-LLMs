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
    
    def __init__(self, policy_model, tokenizer, sampling_params, config_name, reward_model, method, ORM_type):
        super().__init__(policy_model, tokenizer, sampling_params, config_name, reward_model, method, ORM_type)
        
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
        all_step_rewards = []  # Store step rewards for each solution
        if self.method in ['weighted_majority', 'best_of_N']:
            for i in range(len(questions)):
                current_generated_solutions = generated_solutions[i*num_return_sequences:(i+1)*num_return_sequences]
                for solution in current_generated_solutions:
                    step_rewards = self.reward_model.get_value_with_steps(questions[i], Path(solution).steps)
                    reward_value = self.get_reward(step_rewards) # get the scalar reward using ORM_type
                    # If the solution is not a valid solution, set the reward to 0. 
                    # To be more precise, run ablation to see if this is necessary
                    if extract_answer(solution) is None:
                        reward_value = 0
                    rewards.append(reward_value)
                    all_step_rewards.append(step_rewards)
        else: # no need to run reward model for majority voting
            rewards = None
            all_step_rewards = None

        for i in range(len(questions)):
            current_generated_solutions = generated_solutions[i*num_return_sequences:(i+1)*num_return_sequences]
            current_rewards = rewards[i*num_return_sequences:(i+1)*num_return_sequences] if rewards is not None else None
            current_step_rewards = all_step_rewards[i*num_return_sequences:(i+1)*num_return_sequences] if all_step_rewards is not None else None
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
            print(f'current step rewards: {current_step_rewards}')
            print("--------------------------------")

            save_data = {
                "question": questions[i],
                "generated_solutions": current_generated_solutions,
                "selected_solution": selected_solution,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "rewards": current_rewards,
                "step_rewards": current_step_rewards
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
        equal_matrix = [[math_equal(solutions[i], solutions[j]) for j in range(len(solutions))] for i in range(len(solutions))]
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()
    config = load_config(args.config)

    # Override the config with the command line arguments
    config["seed"] = args.seed
    config["model_name"] = args.model_name
    config['config_name'] = config['model_name'].split('/')[-1] + '-Wmajority-' + str(config['num_return_sequences']) + '-seed-' + str(config['seed'])

    apply_config(config) # set up wandb, seed and cuda device
    print(config)

    # load dataset and model
    dataset = load_dataset(config["data_path"])
    dataset = dataset['test']
    model_name = config["model_name"]
    # need to set gpu_memory_utilization to 0.8 to avoid OOM when using qwen2.5b with 16
    model, tokenizer = load_model_with_vllm(model_name, task='auto', tensor_parallel_size=len(config["cuda_device_ids"]), gpu_memory_utilization=0.8)
    tokenizer.pad_token = tokenizer.eos_token 
    reward_model = ValueModel_qwen(device = "auto") # Qwen/Qwen2.5-Math-PRM-7B
    start_problem_index = config.get("begin_problem_index", 0)
    end_problem_index = config.get("end_problem_index", len(dataset)-1)
    dataset = dataset.select(range(start_problem_index, end_problem_index))

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
    )

    inference = MajorityInference(model, tokenizer,sampling_params,
                                  config_name,reward_model,method=config["inference_method"],ORM_type=config["ORM_type"])
    # dataset = dataset.shuffle(seed=42).select(range(100))
    start_time = time.time()
    for i in range(0, len(dataset), batch_size):
        questions = dataset['problem'][i:i+batch_size]
        answers = dataset['solution'][i:i+batch_size]            
        accuracy = inference.inference(system_prompt, questions, answers)
        print("Accuracy: {}".format(accuracy))
        wandb.log({"accuracy": accuracy},step = (i+batch_size))
        # inference.reset()
    end_time = time.time()
    # print(f"generated tokens per sample in average: {np.mean(inference.num_generated_tokens) * num_return_sequences}")
    wandb.log({"generated tokens per sample in average": np.mean(inference.num_generated_tokens) * num_return_sequences})
    print("Time taken: {} seconds".format(end_time - start_time))
    print(f"index range for the dataset: {(start_problem_index, end_problem_index)}")

    wandb.finish()


        
    

