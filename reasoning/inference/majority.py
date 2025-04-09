import os
# set the environment variable for the GPUs to use before importing torc
from reasoning.tools.utils import load_model_with_vllm
from reasoning.evaluator.math_grader import math_equal, extract_answer
from datasets import load_dataset
from vllm import SamplingParams
import wandb
import time
from reasoning.tools.logger import load_config, apply_config
import numpy as np
import json
import argparse

# this contains both weighted and unweighted majority inference
class MajorityInference:
    
    def __init__(self, model, tokenizer, sampling_params, config_name, method = 'majority'):
        self.model = model
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.sample_size = 0
        self.right_count = 0
        self.num_generated_tokens = []
        self.reward = None
        if method not in ['majority', 'weighted_majority', 'best_of_N']:
            raise ValueError('method must be either majority or weighted_majority')
        self.method = method
        
        self.method_dict = {
            'majority': self.majority_vote,
            'weighted_majority': self.weighted_majority_vote,
            'best_of_N': self.best_of_N
        }
        self.config_name = config_name
        results_dir = "/ssdscratch/byuan48/efficient_reasoning/results"
        os.makedirs(results_dir, exist_ok=True)
        self.results_file = os.path.join(results_dir, f"{self.config_name}.jsonl")
        if os.path.exists(self.results_file): # empty the file in the beginning
            os.remove(self.results_file)

    def add_reward(self, reward):
        self.reward = reward
        
    def reset(self):
        self.sample_size = 0
        self.right_count = 0
        self.reward = None
            
    def generate_text(self, system_prompt, questions):
        sampling_params = self.sampling_params
        conversations = [[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ] for question in questions]

        outputs = self.model.chat(conversations, sampling_params)
        texts = []
        num_generated_tokens  = []
        for output in outputs:
            for gen_output in output.outputs:
                texts.append(gen_output.text)
                num_generated_tokens.append(len(gen_output.token_ids))
        # print(texts)
        return texts, num_generated_tokens
    
    def majority_vote(self, generated_text):
        # extract the generated text for a single problem
        solutions = [extract_answer(text) for text in generated_text]
        if len(solutions) == 1:
            return solutions[0]
        # create a matrix to store if any two solutions are equal
        equal_matrix = [[math_equal(solutions[i], solutions[j]) for j in range(i,len(solutions))] for i in range(len(solutions))]
        # majority vote: choose the solution whose number of equal solutions is the largest  (sum of each row)
        # if there are multiple solutions with the same number of equal solutions, choose the first one
        # print(equal_matrix)
        majority_solution = solutions[max(range(len(equal_matrix)), key=lambda i: sum(equal_matrix[i]))]
        return majority_solution
    
    def weighted_majority_vote(self, generated_text):
        solutions = [extract_answer(text) for text in generated_text]
        rewards = self.reward(solutions)
        # to be implemented
        raise NotImplementedError('weighted majority vote is not implemented')
    
    def best_of_N(self, generated_text):
        solutions = [extract_answer(text) for text in generated_text]
        rewards = self.reward(solutions)
        # return the solution with the highest reward, if there are multiple solutions with the same reward, choose the first one
        best_solution = solutions[max(range(len(rewards)), key=lambda i: rewards[i])]
        return best_solution
    
    def inference(self, system_prompt, questions, answers):
        inference_method = self.method_dict[self.method]
        self.sample_size += len(questions)
        num_return_sequences = self.sampling_params.n

        generated_solutions, num_generated_tokens = self.generate_text(system_prompt, questions)
        self.num_generated_tokens.extend(num_generated_tokens)
        for i in range(len(questions)): # input batch implemented in the main function
            current_generated_solutions = generated_solutions[i*num_return_sequences:(i+1)*num_return_sequences]
            selected_solution = inference_method(current_generated_solutions)
            extracted_answer = extract_answer(answers[i])
            is_correct = math_equal(selected_solution, extracted_answer)
            if extracted_answer is None:
                raise ValueError('extracted answer is None')
            if is_correct:
                self.right_count += 1
            mean_num_generated_tokens = np.mean(num_generated_tokens[i*num_return_sequences:(i+1)*num_return_sequences])
            print("--------------------------------")
            print(f'question: {questions[i]}')
            print(f'majority solution: {selected_solution}')
            print(f'extracted answer: {extracted_answer}')
            print(f'is correct: {is_correct}')
            print(f'mean_num_generated_tokens: {mean_num_generated_tokens}')
            print("--------------------------------")
            self.save_solutions_to_jsonl(questions[i], current_generated_solutions, selected_solution, extracted_answer, is_correct, mean_num_generated_tokens)
        print(f'processed {self.sample_size} samples')

        return self.right_count / self.sample_size
    
    def save_solutions_to_jsonl(self, question, current_generated_solutions, selected_solution, extracted_answer, is_correct, mean_num_generated_tokens):
           # Open file in append mode
        with open(self.results_file, 'a') as f:
            result = {
                "question": question,
                "majority_solution": selected_solution,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "generated_solutions": current_generated_solutions,
                "mean_num_generated_tokens": mean_num_generated_tokens
            }
            
            # Write as a single line of JSON
            f.write(json.dumps(result) + '\n')
        
        # # Log to wandb if it's initialized
        # if wandb.run is not None:
        #     # Create artifact
        #     artifact = wandb.Artifact(
        #         name=f"majority_solutions_{os.path.basename(self.results_file)}",
        #         type="solutions",
        #         description="Generated solutions from majority vote inference"
        #     )
            
        #     # Add the file to the artifact
        #     artifact.add_file(self.results_file)
            
        #     # Log the artifact
        #     wandb.log_artifact(artifact)
            
        #     # Also log key statistics
    
    

if __name__ == "__main__":
    
    # model, tokenizer = load_model("gpt2")
    # inference = MajorityInference(model, tokenizer)
    # generated_text = ['\\boxed{1}', '\\boxed{2}', '\\boxed{3}', '\\boxed{4}', '\\boxed{4}']
    # majority_solution = inference.majority_vote(generated_text)
    # print("majority solution: {}".format(majority_solution))

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
    model, tokenizer = load_model_with_vllm(model_name, task='auto', tensor_parallel_size=len(config["cuda_device_ids"]), gpu_memory_utilization=0.9)
    tokenizer.pad_token = tokenizer.eos_token    

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
    ) # shouldn't set seed for random sampling

    inference = MajorityInference(model, tokenizer,sampling_params,config_name,method='majority')
    # make a batch of samples

    start_time = time.time()
    for i in range(0, len(dataset), batch_size):
        questions = dataset['problem'][i:i+batch_size]
        answers = dataset['solution'][i:i+batch_size]            
        accuracy = inference.inference(system_prompt, questions, answers)
        print("Accuracy: {}".format(accuracy))
    inference.reset()
    wandb.log({"accuracy": accuracy})
    end_time = time.time()
    print(f"current majority inference batch size: {num_return_sequences}")
    print(f"total mean num generated tokens: {np.mean(inference.num_generated_tokens)}")
    wandb.log({"total_mean_num_generated_tokens": np.mean(inference.num_generated_tokens)})
    print("Time taken: {} seconds".format(end_time - start_time))

    wandb.finish()

        
    

