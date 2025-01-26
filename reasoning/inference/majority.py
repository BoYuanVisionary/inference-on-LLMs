import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from reasoning.tools.utils import load_model_with_vllm
from reasoning.evaluator.math_grader import math_equal, extract_answer
from datasets import load_dataset
from vllm import SamplingParams
import wandb
import torch.distributed as dist
import time

# this contains both weighted and unweighted majority inference
class MajorityInference:
    
    def __init__(self, model, tokenizer,method = 'majority'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.sample_size = 0
        self.right_count = 0
        self.reward = None
        if method not in ['majority', 'weighted_majority', 'best_of_N']:
            raise ValueError('method must be either majority or weighted_majority')
        self.method = method
        
        self.method_dict = {
            'majority': self.majority_vote,
            'weighted_majority': self.weighted_majority_vote,
            'best_of_N': self.best_of_N
        }
        
    def add_reward(self, reward):
        self.reward = reward
        
    def reset(self):
        self.sample_size = 0
        self.right_count = 0
        self.reward = None
        
    # def generate_text(self, input_text, num_return_sequences, max_new_tokens=1024, top_p=0.9, temperature=0.7):
    #     # input_shape: (batch_size, seq_len), output_shape: (batch_size, seq_len + max_new_tokens)
    #     # note that when using eos_token_id in the batch generation, the eos_token will be added to the end of the sequence but acutally no inference needed
    #     tokens = self.tokenizer(input_text, return_tensors = "pt", padding=True, truncation=True, max_length=1024).to(self.device)
    #     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_flops=True) as prof:
    #         with torch.no_grad():
    #             res = self.model.generate(
    #                 **tokens,
    #                 max_new_tokens = max_new_tokens,
    #                 do_sample = True,
    #                 eos_token_id = [self.tokenizer.eos_token_id],
    #                 num_return_sequences = num_return_sequences,
    #                 top_p = top_p,
    #                 temperature = temperature
    #             )
    #     print(prof.key_averages().table(sort_by="flops"))
    #     # print('questions in this batch: {}'.format(input_text))
    #     # print('generated solutions: {}'.format(self.tokenizer.batch_decode(res, skip_special_tokens=True)))
    #     return self.tokenizer.batch_decode(res, skip_special_tokens=True)
    
    def generate_text(self, input_text, num_return_sequences, max_new_tokens=1024, top_p=0.9, temperature=0.7):
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=num_return_sequences,
            stop_token_ids=[self.tokenizer.eos_token_id],
            skip_special_tokens = True,
            include_stop_str_in_output = False
        )
        outputs = self.model.generate(input_text, sampling_params)
        texts = []
        for output in outputs:
            for gen_output in output.outputs:
                texts.append(gen_output.text)
        # print(texts)
        return texts
    
    def majority_vote(self, generated_text):
        # extract the generated text for a single problem
        solutions = [extract_answer(text) for text in generated_text]
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
    
    def inference(self, questions, answers, num_return_sequences):
        inference_method = self.method_dict[self.method]
        self.sample_size += len(questions)
        generated_solutions = self.generate_text(questions,num_return_sequences)
        for i in range(len(questions)):
            selected_solution = inference_method(generated_solutions[i*num_return_sequences:(i+1)*num_return_sequences])
            extracted_answer = extract_answer(answers[i])
            if extracted_answer is None:
                raise ValueError('extracted answer is None')
            if math_equal(selected_solution, extracted_answer):
                self.right_count += 1
            print(f'question: {questions[i]}')
            print(f'majority solution: {selected_solution}')
            print(f'extracted answer: {extracted_answer}')
            print(math_equal(selected_solution, extracted_answer))
            print("--------------------------------")
        print(f'processed {self.sample_size} samples')
        return self.right_count / self.sample_size
    

if __name__ == "__main__":
    
    # model, tokenizer = load_model("gpt2")
    # inference = MajorityInference(model, tokenizer)
    # generated_text = ['\\boxed{1}', '\\boxed{2}', '\\boxed{3}', '\\boxed{4}', '\\boxed{4}']
    # majority_solution = inference.majority_vote(generated_text)
    # print("majority solution: {}".format(majority_solution))
    
    wandb.init(project="efficient_reasoning")
    wandb.config.update({"model": "Llama-3.2-3B-Instruct", "batch_size": 10})
    # batch size does not make a huge difference here

    dataset = load_dataset("HuggingFaceH4/MATH-500")
    dataset = dataset['test']
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model, tokenizer = load_model_with_vllm(model_name)
    # set pad token to eos token
    tokenizer.pad_token = tokenizer.eos_token
    inference = MajorityInference(model, tokenizer, method='majority')
    
    batch_size = 10
    # make a batch of samples
    for num_return_sequences in [2,4,8,16,32,64,128]:
        wandb.log({"num_return_sequences": num_return_sequences})
        start_time = time.time()
        for i in range(0, len(dataset), batch_size):
            questions = dataset['problem'][i:i+batch_size]
            answers = dataset['solution'][i:i+batch_size]
            accuracy = inference.inference(questions, answers, num_return_sequences=num_return_sequences)
            print("Accuracy: {}".format(accuracy))
        inference.reset()
        wandb.log({"accuracy": accuracy})
        end_time = time.time()
        print(f"current majority inference batch size: {num_return_sequences}")
        print("Time taken: {} seconds".format(end_time - start_time))
    wandb.finish()
        
    if dist.is_initialized():
        dist.destroy_process_group()
        
    
    

