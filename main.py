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
full_dataset = load_dataset("lighteval/MATH", split="test")
print(f"Original Dataset Size: {len(full_dataset)}")

# Generate a 20% subset of the dataset
shuffled_dataset = full_dataset.shuffle(seed=42)
dataset = shuffled_dataset.train_test_split(test_size=0.2, seed=42)["test"]
print(f"Subset Size: {len(dataset)}")
print(dataset[0])

    
parser = argparse.ArgumentParser(description="model name")
parser.add_argument("-model", type=str, help="model name")
args = parser.parse_args()
# Load the LLaMA 3 8B model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct" if args.model == '8' else "meta-llama/Llama-3.2-3B-Instruct"
device = torch.device("cuda:1") if model_name == "meta-llama/Llama-3.2-3B-Instruct" else torch.device("cuda:1")
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

def generate_batch(model, tokenizer, input_texts, number_of_samples, max_new_tokens):
    """Generates responses for a batch of input texts."""
    inputs = tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)
    log_all_token_probs = torch.zeros([len(input_texts), number_of_samples,max_new_tokens]) + 100
    print(f'running model.generate for {len(input_texts)} inputs')
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
    print(f'finished model.generate for {len(input_texts)} inputs')
    batch_solutions = []
    batch_probabilities = []
    
    for j, input_text in enumerate(input_texts):  # Process each input in the batch
        solutions = []
        probabilities = []
        for i in range(number_of_samples):
            idx = j * number_of_samples + i
            output = output_.sequences[idx].tolist()[len(inputs["input_ids"][j]):]
            # Find the position of the EOS token and truncate if it exists
            if tokenizer.eos_token_id in output:
                eos_position = output.index(tokenizer.eos_token_id)
                output = output[:eos_position]
            response = tokenizer.decode(output)
            
            scores = output_.scores

            log_token_probs = [score[idx].log_softmax(dim=-1)[token].item() for score, token in zip(scores, output)]
            log_all_token_probs[j, i, :len(log_token_probs)] = torch.tensor(log_token_probs)
            sentence_log_prob = sum(log_token_probs) / len(log_token_probs) if len(log_token_probs) > 0 else None
            
            solutions.append(response)
            probabilities.append(sentence_log_prob)
        
        batch_solutions.append(solutions)
        batch_probabilities.append(probabilities)
    print(f'finished processing {len(input_texts)} inputs')
    return batch_solutions, batch_probabilities, log_all_token_probs


# Initialize a wandb run
# wandb.init(project="math-dataset-evaluation", name = model_name)

system_prompt = '''
You are a helpful assistant for math problem-solving. At the end of the solution, provide the final answer in the format: \boxed{answer}. 
Now solve the following problem:'''
correct = 0
number_invalid_generated_answers = 0

number_of_questions = len(dataset)
batch_size = 2 
best_of_n = 8
max_new_tokens = 750 if model_name == "meta-llama/Llama-3.2-8B-Instruct" else 1024
log_probs = torch.zeros([number_of_questions, best_of_n, max_new_tokens]) + 100

for start_idx in range(0, number_of_questions, batch_size):
    end_idx = min(start_idx + batch_size, number_of_questions)
    indices = range(start_idx, end_idx)
    
    questions, solutions = select_questions_batch(dataset, indices)
    input_texts = [system_prompt + ' ' + question for question in questions]
    
    batch_solutions, batch_log_probabilities, batch_log_probs = generate_batch(model, tokenizer, input_texts, best_of_n, max_new_tokens)
    save_batch_solutions_to_jsonl(batch_solutions, input_texts, model_name.split('/')[-1]+'_raw'+'.jsonl')
    log_probs[start_idx:end_idx] = batch_log_probs
    generated_answers, number_invalid_generated_answers_batch = extract_boxed_answers(batch_solutions)
    number_invalid_generated_answers += number_invalid_generated_answers_batch
    GT_answers, number_invalid_GT_answers = extract_boxed_answers([[sol] for sol in solutions]) 
    save_batch_solutions_to_jsonl(generated_answers, GT_answers, model_name.split('/')[-1]+'_answer'+'.jsonl')
    if number_invalid_GT_answers > 0:
        print('*' * 50)
        print('\n')
        print(f"Invalid ground truth answers found in batch {start_idx}-{end_idx}")
        print('*' * 50)
    
    for i, (answers, GT) in enumerate(zip(generated_answers, GT_answers)):
        print(f"Question_index: {start_idx + i}")
        print(f"Generated Answers: {answers}")
        print(f"Log Probabilities: {batch_log_probabilities[i]}")
        print(f"Ground Truth Solution: {solutions[i]}")
        print('\n')
        
        # Log each result to wandb
        # wandb.log({
        #     "question_index": start_idx + i,
        #     "question": questions[i],
        #     "ground_truth": GT[0],
        #     "generated_answers": answers,
        #     "log_probabilities": batch_log_probabilities[i],
        # })
        
        # Update the correct counter
        # correct += 1 if check_equivalence(answers, GT[0]) else 0
        correct += 1 if GT[0] in answers else 0
        
    if start_idx % 100 == 0:
        print(f"Progress: {end_idx}/{number_of_questions}")
        print(f"Accuracy: {correct / (end_idx)}")
        print(f"Invalid generated answers: {number_invalid_generated_answers}, overall answers: {(end_idx)*best_of_n}")

# Save the log probabilities to a file
torch.save(log_probs, model_name.split('/')[-1] + "_log_probs.pt")


# Calculate final accuracy
accuracy = correct / number_of_questions
print(f'Accuracy: {accuracy}')

# Log final accuracy to wandb
# wandb.log({"accuracy": accuracy})

# Finish the wandb run
# wandb.finish()