from reasoning.tools.utils import  load_model_with_vllm
import torch
from vllm import SamplingParams
import requests
import json



# Don't find reliable ways to load model with vllm on specific GPUs. 
# For now, just use cuda_visible_devices and assume both reward model and proposal models use all the visible GPUs.

# to do: check if it works in the mcts task
# class LlamaPolicy:
#     def __init__(self, model_name, 
#                  temperature=0.7, top_p=0.9, 
#                  max_new_tokens=128, do_sample=True, gpu_memory_utilization=1):
#         self.model, self.tokenizer = load_model_with_vllm(model_name, task = 'auto', 
#                                                           tensor_parallel_size = self.__gpu_count(), gpu_memory_utilization=gpu_memory_utilization)
#         self.temperature = temperature
#         self.top_p = top_p
#         self.max_new_tokens = max_new_tokens
#         self.do_sample = do_sample
        
#         self.sampling_params = SamplingParams(
#             temperature=self.temperature,
#             top_p=self.top_p,
#             max_tokens=self.max_new_tokens,
#             stop_token_ids=[self.tokenizer.eos_token_id,
#                             self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
#             skip_special_tokens=False,
#         )
        
#     def __gpu_count(self):
#         return torch.cuda.device_count()
    
#     def get_proposal(self, prompt, n_responses):
#         return self.get_local_response_llama_vllm(prompt, n_responses)
        
#     def get_local_response_llama_vllm(self, query,n_responses): # batch generation
        
#         cnt = 2
#         split_response = [[''] for _ in range(n_responses)]
#         self.sampling_params.n = n_responses
#         message = '<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'.format(query=query)
#         while cnt:
#             try:
#                 outputs = self.model.generate(message, sampling_params=self.sampling_params)
#                 split_response = [outputs[0].outputs[i].text.strip().split('\n\n') for i in range(n_responses)]
#                 break
#             except Exception as e:
#                 print(f'Error:{e}, obtain response again...\n')
#                 cnt -= 1

#         return split_response

#     def test(self):
#         # test the model with a simple query to see if the format is correct
#         query = r"""Given a science problem and an existing incomplete solution, your task is to complete the solution in a smooth and proper way.

#             - If no existing steps are provided, you must briefly analyze the problem and output only the first step.  
#             - If existing steps are sufficent to solve the problem, you must output the final answer and format the answer inside `\\boxed{}` (e.g., `\\boxed{42}`). 
#             - If existing steps are not sufficent to solve the problem, you must output exactly **one** correct next step that naturally follows from the previous ones.  
#             - You **must** follow the given format.

#             **Strict Output Format:**  
#             - Your response must always start with: `Next step: ...`  
#             - The response must be limited to one reasoning step (e.g., a calculation, reasoning, or answer choice).  

#             If there are multiple reasonable next steps, choose the most natural one based on the provided existing steps.  

#             Here is the problem and the existing steps:  

#             Problem: How many positive whole-number divisors does 196 have?
#             Existing Steps:
#             Step 1: Factorize 196 into its prime factors to determine its divisors. To find the number of positive whole-number divisors of 196, we first need to factorize 196 into its prime factors.  ##
#             Step 2: The prime factorization of 196 is $2^2 \cdot 7^2$.

#             Output:"""
#         response = self.get_proposal(query, n_responses=1)
#         print(f'response: {response}')

class QwenPolicy:
    def __init__(self, model_name, 
                 temperature=0.7, top_p=0.9, 
                 max_new_tokens=512, do_sample=True, gpu_memory_utilization=1):
        self.model, self.tokenizer = load_model_with_vllm(model_name, task = 'auto', 
                                                          tensor_parallel_size = self.__gpu_count(), gpu_memory_utilization=gpu_memory_utilization)
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            skip_special_tokens=False,
        )
        
    def __gpu_count(self):
        return torch.cuda.device_count()
    
    def get_proposal(self, system_prompt, query, n_responses=1, partial_solutions=None):
        if partial_solutions is None:
            return self.get_local_response_llama_vllm(system_prompt, query, n_responses)
        else:
            return self.get_local_response_llama_vllm_completion(system_prompt, query, partial_solutions, n_responses)
        
    def get_local_response_llama_vllm(self, system_prompt, query, n_responses): # batch generation
        
        cnt = 2
        self.sampling_params.n = n_responses
        conversations = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ] 
        while cnt:
            try:
                outputs = self.model.chat(conversations, sampling_params=self.sampling_params)
                split_response = [outputs[0].outputs[i].text for i in range(n_responses)]
                break
            except Exception as e:
                print(f'Error:{e}, obtain response again...\n')
                cnt -= 1

        return split_response
    
    def get_local_response_llama_vllm_completion(self, system_prompt, query, partial_solutions, n_responses): # batch generation
        
        cnt = 2
        self.sampling_params.n = n_responses
        conversations = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": partial_solutions
            }
        ] 
        while cnt:
            try:
                outputs = self.model.chat(conversations, sampling_params=self.sampling_params)
                split_response = [outputs[0].outputs[i].text for i in range(n_responses)]
                break
            except Exception as e:
                print(f'Error:{e}, obtain response again...\n')
                cnt -= 1

        return split_response

    def test(self):

        system_prompt = r"""Given a science problem and an existing incomplete solution, your task is to complete the solution in a smooth and proper way.

            - If no existing steps are provided, you must briefly analyze the problem and output only the first step.  
            - If existing steps are sufficent to solve the problem, you must output the final answer and format the answer inside `\\boxed{}` (e.g., `\\boxed{42}`). 
            - If existing steps are not sufficent to solve the problem, you must output exactly **one** correct next step that naturally follows from the previous ones.  
            - You **must** follow the given format.

            **Strict Output Format:**  
            - Your response must always start with: `Next step: ...`  
            - The response must be limited to one reasoning step (e.g., a calculation, reasoning, or answer choice).  

            If there are multiple reasonable next steps, choose the most natural one based on the provided existing steps. """
        
        query = r""" 
            Here is the problem and the existing steps:  

            Problem: How many positive whole-number divisors does 196 have?
            Existing Steps:
            Null

            Output:"""
        response = self.get_proposal(system_prompt,query, n_responses=1)
        print(f'response: {response}')
    

class QwenReward:
    def __init__(self, model_name, 
                 temperature=0.7, top_p=0.9, 
                 max_new_tokens=128, do_sample=True, gpu_memory_utilization=1,low=0):
        self.model, self.tokenizer = load_model_with_vllm(model_name, task = 'auto', 
                                                          tensor_parallel_size = self.__gpu_count(), gpu_memory_utilization=gpu_memory_utilization)
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            skip_special_tokens=False,
        )
        
        self.system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        self.low = low
        
    def __gpu_count(self):
        return torch.cuda.device_count()
    
    def prepare_input(self, query, response_steps):
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "<extra_0>".join(response_steps) + "<extra_0>"},
        ]
        
        conversation_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        input_ids = self.tokenizer.encode(conversation_str, return_tensors="pt").to(self.model.device)
        token_masks = (input_ids == self.tokenizer.encode("<extra_0>")[0])
        
        return input_ids, token_masks
    
    def compute_rewards(self, input_ids, token_masks):
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        
        logits = outputs[0]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        
        batch_rewards = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            batch_rewards.append(positive_probs.cpu().tolist())
            
        return batch_rewards
    
    def format_steps(self,input_text):

        steps = []
        current_step = []
        
        for line in input_text.split('\n'):
            line = line.strip()
            if line.startswith('Step') and ':' in line:
                
                if current_step:
                    steps.append(' '.join(current_step))
                    current_step = []
                step_header, _, content = line.partition(':')
                current_step.append(f"{step_header.strip()}:")
                if content:
                    current_step.append(content.strip())
            elif line and current_step:     
                current_step.append(line.strip())
        if current_step:
            steps.append(' '.join(current_step))
         
        formatted_steps = []
        for i, step in enumerate(steps, 1):
            step = step.replace(f"Step {i}:", "")
            step = step.replace('##', '').strip()
            formatted_steps.append(step)
            
        return formatted_steps
                
    def get_value(self, query, output):
        try:
            formatted_steps = self.format_steps(output)
            input_ids, token_masks = self.prepare_input(query, formatted_steps)
            step_rewards = self.compute_rewards(input_ids, token_masks)
            return step_rewards[0][-1] # last step reward is better
        except Exception as e:
            print(f"Error in get_value: {str(e)}")
            return self.low




        
        

