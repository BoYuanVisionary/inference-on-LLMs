import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import re
import wandb
import warnings
from datasets import load_dataset
import argparse

from utils import extract_boxed_answers, check_equivalence, save_batch_solutions_to_jsonl

# Load the math dataset
full_dataset = load_dataset("lighteval/MATH", split="train")
print(f"Original Dataset Size: {len(full_dataset)}")
dataset = full_dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.0002, seed=42)["test"]
    
parser = argparse.ArgumentParser(description="model name")
parser.add_argument("-device", type=int, help="device")
args = parser.parse_args()
device = torch.device(args.device)
model_name =  "meta-llama/Llama-3.2-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_cache=False,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map = device,
)
model.config.use_flash_attention = True

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' # to prevent errors with FA
tokenizer.truncation_side = 'left' # to prevent cutting off last generation


def select_questions_batch(dataset, indices):
    """Selects a batch of questions from the dataset by indices."""
    questions = [dataset[i]["problem"] for i in indices]
    solutions = [dataset[i]["solution"] for i in indices]
    return questions, solutions


def generate_single_solution(model, tokenizer, input_text, max_new_tokens):
    """Generates a single response for a given input."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)
    output = model.generate(
        **inputs,
        do_sample=True, 
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        return_dict_in_generate=True,
        output_scores=False,
        use_cache=True,
    )
    response = tokenizer.decode(output.sequences[0], skip_special_tokens=False)
    return response

def generate_batch(model, tokenizer, input_texts, number_of_samples, max_new_tokens):
    """Generates responses for a batch of input texts."""
    inputs = tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)
    output_ = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=0.9,
        num_return_sequences=number_of_samples,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
    )
    
    scores = output_.scores
    new_genarations = output_[,:]
    return output_, scores
    
    batch_solutions = []
    batch_probabilities = []
    
    # for j, input_text in enumerate(input_texts):  # Process each input in the batch
    #     solutions = []
    #     probabilities = []
    #     for i in range(number_of_samples):
    #         idx = j * number_of_samples + i
    #         output = output_.sequences[idx].tolist()[len(inputs["input_ids"][j]):]
    #         # Find the position of the EOS token and truncate if it exists
    #         if tokenizer.eos_token_id in output:
    #             eos_position = output.index(tokenizer.eos_token_id)
    #             output = output[:eos_position]
    #         response = tokenizer.decode(output)
            
    #         scores = output_.scores

    #         log_token_probs = [score[idx].log_softmax(dim=-1)[token].item() for score, token in zip(scores, output)]
    #         log_all_token_probs[j, i, :len(log_token_probs)] = torch.tensor(log_token_probs)
    #         sentence_log_prob = sum(log_token_probs) / len(log_token_probs) if len(log_token_probs) > 0 else None
            
            
    #         entropy_values = []
        
    #         for score in scores:
    #             prob_dist = score.softmax(dim=-1)  
    #             entropy = -torch.sum(prob_dist * prob_dist.log(), dim=-1).item()  # Compute entropy
    #             entropy_values.append(entropy)
        
        
    #         entropy_all_token[j, i, :len(entropy_values)] = torch.tensor(entropy_values)
            
    #         solutions.append(response)
    #         probabilities.append(sentence_log_prob)
        
    #     batch_solutions.append(solutions)
    #     batch_probabilities.append(probabilities)
    
    return batch_solutions, batch_probabilities, log_all_token_probs, entropy_all_token


number_of_questions = len(dataset)

# Modified loop
system_prompt = '''
You are a helpful assistant for math problem-solving. At the end of the solution, provide the final answer in the format: \boxed{answer}. 
Now solve the following problem:'''
batch_size = 2
best_of_n = 2  # Number of samples for complete answers
max_new_tokens = 50  # For initial single solution generation
max_new_tokens_for_complete = 100  # For generating n complete answers

for start_idx in range(0, number_of_questions, batch_size):
    end_idx = min(start_idx + batch_size, number_of_questions)
    indices = range(start_idx, end_idx)
    
    questions, solutions = select_questions_batch(dataset, indices)
    input_texts = [system_prompt + ' ' + question for question in questions]
    
    for question_idx, question in enumerate(input_texts):
        print(f"\033[31m Processing question {start_idx + question_idx}")
        
        # Step 1: Generate single solution
        single_response = generate_single_solution(model, tokenizer, question, max_new_tokens)
        print(f"\033[32m Single Solution: {single_response}")
        
        # Step 2: Generate n complete answers
        complete_answers = generate_n_complete_answers(model, tokenizer, question, single_response, best_of_n, max_new_tokens_for_complete)
        print(f"\033[33m Generated Complete Answers: {complete_answers}")
        
        # Save or evaluate as needed
        # save_batch_solutions_to_jsonl([complete_answers], [question], model_name.split('/')[-1] + '_complete_answers.jsonl')