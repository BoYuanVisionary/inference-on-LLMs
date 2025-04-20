# load a jsonl file
import json
import os
import wandb
import numpy as np
from reasoning.evaluator.math_grader import math_equal, extract_answer
import argparse
import random

def compute_accuracy(step_rewards, ground_truths, generated_answers_lists, 
                    score_method='min', selection_method='majority',
                    math_equal_fn=math_equal):
    """
    Compute accuracy for multiple reasoning solutions using different scoring and selection methods.
    """
    def compute_solution_score(rewards):
        if score_method == 'min':
            return min(rewards)
        elif score_method == 'last':
            return rewards[-1]
        elif score_method == 'product':
            score = 1.0
            for r in rewards:
                score *= r
            return score
        elif score_method == 'geometric_mean':
            return np.prod(rewards) ** (1 / len(rewards))
        else:
            raise ValueError(f"Unknown score method: {score_method}")
    
    def select_answer(answers, scores):
        if len(answers) == 1:
            return answers[0]
            
        if selection_method == 'majority':
            # Create equality matrix
            equal_matrix = [[math_equal_fn(answers[i], answers[j]) 
                           for j in range(len(answers))] 
                           for i in range(len(answers))]
            # Choose solution with most equals
            return answers[max(range(len(equal_matrix)), 
                            key=lambda i: sum(equal_matrix[i]))]
            
        elif selection_method == 'best_n':
            # Simply return solution with highest score
            return answers[max(range(len(scores)), 
                            key=lambda i: scores[i])]
            
        elif selection_method == 'weighted_majority':
            # Initialize weighted votes
            weighted_votes = [0.0] * len(answers)
            
            # Add scores for equal solutions
            for i in range(len(answers)):
                for j in range(len(answers)):
                    if math_equal_fn(answers[i], answers[j]):
                        weighted_votes[i] += scores[j]
            
            # Return solution with highest weighted vote
            return answers[max(range(len(weighted_votes)), 
                            key=lambda i: weighted_votes[i])]
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
    
    correct_count = 0
    total_count = len(ground_truths)
    
    for prob_idx in range(total_count):
        solution_scores = [compute_solution_score(rewards) 
                         for rewards in step_rewards[prob_idx]]
        
        prob_answers = generated_answers_lists[prob_idx]
        
        # Select final answer using specified method
        selected_answer = select_answer(prob_answers, solution_scores)
        
        # Compare with ground truth
        if selected_answer is not None and math_equal_fn(selected_answer, ground_truths[prob_idx]):
            correct_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0

def downsampling(step_rewards, generated_answers_lists, num_samples, seed=42):
    """
    Randomly downsample the solutions and their corresponding rewards.
    
    Args:
        step_rewards: List[List[float]] - rewards for each step for each problem
        generated_answers_lists: List[List[str]] - generated answers for each problem
        num_samples: int - number of samples to keep
        seed: int - random seed for reproducibility
    
    Returns:
        tuple: (downsampled_step_rewards, downsampled_generated_answers_lists)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    downsampled_step_rewards = []
    downsampled_generated_answers_lists = []
    
    for i in range(len(step_rewards)):
        # Get total number of available samples
        n_available = len(step_rewards[i])
        
        # If we request more samples than available, use all samples
        if num_samples >= n_available:
            downsampled_step_rewards.append(step_rewards[i])
            downsampled_generated_answers_lists.append(generated_answers_lists[i])
            continue
            
        # Generate random indices without replacement
        selected_indices = random.sample(range(n_available), num_samples)
        
        # Sample the rewards and answers using these indices
        sampled_rewards = [step_rewards[i][idx] for idx in selected_indices]
        sampled_answers = [generated_answers_lists[i][idx] for idx in selected_indices]
        
        downsampled_step_rewards.append(sampled_rewards)
        downsampled_generated_answers_lists.append(sampled_answers)
    
    return downsampled_step_rewards, downsampled_generated_answers_lists

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama-3.2-3B-Instruct")
    parser.add_argument("--method", type=str, default="Wmajority")
    parser.add_argument("--num_generations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_name = args.model_name
    method = args.method
    num_generations = args.num_generations
    seed = args.seed

    results_folder = '/ssdscratch/byuan48/efficient_reasoning/results'
    results_file  = f"{model_name}-{method}-{num_generations}-seed-{seed}.jsonl"
    with open(os.path.join(results_folder, results_file), 'r') as f:
        data_list = [json.loads(line) for line in f]
    step_rewards = [item['step_rewards'] for item in data_list]
    ground_truths = [item['extracted_answer'] for item in data_list]
    generated_answers_lists = []
    for item in data_list:
        generated_answers_list = [] 
        for generated_solution in item['generated_solutions']:
            generated_answers_list.append(extract_answer(generated_solution))
        generated_answers_lists.append(generated_answers_list)

    print(f'{model_name} {method} {num_generations} {seed}')
    wandb.init(project="efficient_reasoning", name=f'analysis on {model_name} {method} {num_generations} {seed}')
    for num_samples in [2,4,8,16,32]:
        for seed in [0,1,2]:
            for score_method in [ 'product']:
                for selection_method in [ 'weighted_majority']:
                    downsampled_step_rewards, downsampled_generated_answers_lists = downsampling(step_rewards, generated_answers_lists, num_samples, seed)
                    accuracy = compute_accuracy(downsampled_step_rewards, ground_truths, downsampled_generated_answers_lists,
                        score_method=score_method, selection_method=selection_method,
                        math_equal_fn=math_equal)
                    print(f"Accuracy: {accuracy} for seed {seed}, num_samples {num_samples}, score_method {score_method}, selection_method {selection_method}")
                    wandb.log({
                        "accuracy": accuracy,
                        "seed": seed,
                        "num_samples": num_samples,
                        "score_method": score_method,
                        "selection_method": selection_method
                    })
    
    # If number_samples is 64, then just run the function once
    accuracy = compute_accuracy(step_rewards, ground_truths, generated_answers_lists,
                        score_method='product', selection_method='weighted_majority',
                        math_equal_fn=math_equal)
    print(f"Accuracy: {accuracy} for seed {seed}, num_samples 64, score_method product, selection_method weighted_majority")
    wandb.log({
        "accuracy": accuracy,
        "seed": seed,
        "num_samples": 64,
        "score_method": 'product',
        "selection_method": 'weighted_majority'
    })
    wandb.finish()
