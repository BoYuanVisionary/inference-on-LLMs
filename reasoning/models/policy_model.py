from reasoning.tools.utils import  load_model_with_vllm
import torch
from vllm import SamplingParams
from abc import ABC, abstractmethod
# Don't find reliable ways to load model with vllm on specific GPUs. 
# For now, just use cuda_visible_devices and assume both reward model and proposal models use all the visible GPUs.

# to do: check if it works in the mcts task

class base_policy_model(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def get_proposal(self, prompt, n_responses):
        pass    
    
    def _gpu_count(self):
        return torch.cuda.device_count()
    

class LlamaPolicy(base_policy_model):
    def __init__(self, model_name, 
                 temperature=0.7, top_p=0.9, 
                 max_new_tokens=128, do_sample=True, gpu_memory_utilization=1):
        super().__init__()
        self.model, self.tokenizer = load_model_with_vllm(model_name, task = 'auto', 
                                                          tensor_parallel_size = self._gpu_count(), gpu_memory_utilization=gpu_memory_utilization)
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
                    
    def get_proposal(self, query, n_responses): # batch generation
        
        cnt = 2
        split_response = [[''] for _ in range(n_responses)]
        self.sampling_params.n = n_responses
        message = '<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'.format(query=query)
        while cnt:
            try:
                outputs = self.model.generate(message, sampling_params=self.sampling_params)
                split_response = [outputs[0].outputs[i].text.strip().split('\n\n') for i in range(n_responses)]
                break
            except Exception as e:
                print(f'Error:{e}, obtain response again...\n')
                cnt -= 1

        return split_response
    

class LlamaPolicy(base_policy_model):
    def __init__(self, model_name, 
                 temperature=0.7, top_p=0.9, 
                 max_new_tokens=128, do_sample=True, gpu_memory_utilization=1):
        super().__init__()
        self.model, self.tokenizer = load_model_with_vllm(model_name, task = 'auto', 
                                                          tensor_parallel_size = self._gpu_count(), gpu_memory_utilization=gpu_memory_utilization)
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
                    
    def get_proposal(self, query, n_responses): # batch generation
        
        cnt = 2
        split_response = [[''] for _ in range(n_responses)]
        self.sampling_params.n = n_responses
        message = '<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'.format(query=query)
        while cnt:
            try:
                outputs = self.model.generate(message, sampling_params=self.sampling_params)
                split_response = [outputs[0].outputs[i].text.strip().split('\n\n') for i in range(n_responses)]
                break
            except Exception as e:
                print(f'Error:{e}, obtain response again...\n')
                cnt -= 1

        return split_response





