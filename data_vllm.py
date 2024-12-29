import numpy as np
from datasets import load_dataset
import time
import argparse
import json
from vllm import LLM, SamplingParams
import gc
import torch
import torch.distributed as dist

# use argparse to specify the range of questions to generate
parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=0, help="beginning index of the dataset")
args = parser.parse_args()

# Load the math dataset from Hugging Face
dataset = load_dataset("lighteval/MATH", split="train", trust_remote_code=True)
print(dataset)
print(dataset[0])
# randomly select a subset from the dataset
dataset = dataset.shuffle(seed=4)
dataset = dataset.select(range(args.index*10,(args.index+1)*10))

def select_questions_batch(dataset, indices):
    """Selects a batch of questions from the dataset by indices."""
    questions = [dataset[i]["problem"] for i in indices]
    solutions = [dataset[i]["solution"] for i in indices]
    return questions, solutions


# Initialize the model
model_name = "meta-llama/Llama-3.2-3B-Instruct"
llm = LLM(model=model_name,tensor_parallel_size=1,gpu_memory_utilization=0.9)
tokenizer = llm.get_tokenizer()

def return_entropy(logprobs):
    # Calculate entropy for each token, return a list of entropies
    entropies = []
    num_tokens = len(logprobs) # number of tokens
    for i in range(num_tokens):
        temp = []
        for token_id in logprobs[i].keys():
            temp.append(logprobs[i][token_id].logprob)
        temp = np.array(temp)
        entropy = -1 * np.sum(np.exp(temp) * temp)
        entropies.append(entropy)
    return entropies

def return_probs(logprobs):
    # Calculate entropy for each token, return a list of probs
    probs = []
    num_tokens = len(logprobs) # number of tokens
    for i in range(num_tokens):
        token_id = next(iter(logprobs[i]))
        probs.append(logprobs[i][token_id].logprob)
    return probs

# Define parameters

n1 = 8  # Number of results for each prompt in Step 1
n2 = 16  # Number of free generations for each result in Step 2
max_tokens_step1 = 256
max_tokens_step2 = 1024
batch_size = 64 

time_step1_begin = time.time()
# Initialize storage for results
all_results_step1 = []
num_batches = (len(dataset) + batch_size - 1) // batch_size
# Process dataset in batches
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(dataset))
    indices = list(range(start_idx, end_idx))

    prompts, solutions = select_questions_batch(dataset, indices)
    system_prompt = "You are a helpful assistant for math problem-solving. At the end of the solution, provide the final answer in the format: \\boxed{answer}. Now solve the following problem: "
    prompts = [system_prompt + prompt for prompt in prompts]

    # Step 1: Generate limited responses with logprobs
    sampling_params_step1 = SamplingParams(
        temperature=0.9,
        max_tokens=max_tokens_step1,
        n=n1,
        stop=["Step 3"],
        logprobs=20,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        skip_special_tokens = False,
        include_stop_str_in_output = True
    )

    outputs_step1 = llm.generate(prompts, sampling_params_step1)

    # Store results for Step 1
    with open(f"./generated_datasets/step1_large_train_{args.index}.jsonl", "a") as f:
            for output in outputs_step1:
                for gen_output in output.outputs:
                    result = {
                        "original_prompt": output.prompt,
                        "original_prompt_tokens": output.prompt_token_ids,
                        "generated_text": tokenizer.decode(gen_output.token_ids, skip_special_tokens=False),
                        "generated_tokens": gen_output.token_ids,
                        "entropy": return_entropy(gen_output.logprobs),
                        "logprobs": return_probs(gen_output.logprobs),
                    }
                    f.write(json.dumps(result) + "\n")

    print(f"Step 1 Batch {batch_idx + 1}/{num_batches} complete!")

print("Step 1 complete! ")
time_step1_end = time.time()
print(f"Time taken for Step 1: {time_step1_end - time_step1_begin:.2f}s")

time_step2_begin = time.time()
# Step 2: Free-generation for each result of Step 1
all_results_step2 = []
sampling_params_step2 = SamplingParams(
    temperature=0.9,
    max_tokens=max_tokens_step2,
    n=n2,
    # seed=42,  # wrong, this will always return the same result even when n > 1
    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    logprobs=20,
)

all_results_step1 = []
with open(f"./generated_datasets/step1_large_train_{args.index}.jsonl", "r") as f:
    for line in f:
        all_results_step1.append(json.loads(line))
combined_tokens_batches = [step1_result["original_prompt_tokens"][1:] + list(step1_result["generated_tokens"]) for step1_result in all_results_step1] # remove the BOS token
combined_text_batches = tokenizer.batch_decode(combined_tokens_batches, skip_special_tokens=False)
batch_size = 32
num_batches = (len(combined_text_batches) + batch_size - 1) // batch_size
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(combined_text_batches))
    indices = list(range(start_idx, end_idx))

    combined_text_batch = combined_text_batches[start_idx:end_idx]
    outputs_step2 = llm.generate(combined_text_batch, sampling_params_step2)
    # Store results for Step 2
    # Save results for Step 2 incrementally
    with open(f"./generated_datasets/step2_large_train_{args.index}.jsonl", "a") as f:
        for output in outputs_step2:
            for gen_output in output.outputs:
                result = {
                    "first_step_text": output.prompt,
                    "first_step_tokens": output.prompt_token_ids,
                    "generated_text": gen_output.text,
                    "generated_tokens": gen_output.token_ids,
                    "entropy": return_entropy(gen_output.logprobs),
                    "logprobs": return_probs(gen_output.logprobs),
                }
                f.write(json.dumps(result) + "\n")
                # delete result to free up memory
                del result
    print(f"Step 2 Batch {batch_idx + 1}/{num_batches} complete!")
    
    # # free up memory
    # gc.collect()  
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()     
    #     torch.cuda.synchronize()  


print("Batch inference complete! ")
time_step2_end = time.time()
print(f"Time taken for Step 2: {time_step2_end - time_step2_begin:.2f}s")

if dist.is_initialized():
    dist.destroy_process_group()
