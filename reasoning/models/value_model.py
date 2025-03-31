from reasoning.tools.utils import  load_model_with_vllm
import torch
from vllm import SamplingParams
from abc import ABC, abstractmethod 


class base_value_model(ABC):
    def __init__(self):
        
        self.low = 0
    
    def _gpu_count(self):
        return torch.cuda.device_count()
    
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
    
    @abstractmethod
    def get_value(self, query, output):
        pass
    @abstractmethod
    def prepare_input(self, query, response_steps):
        pass
    @abstractmethod
    def compute_rewards(self, input_ids, token_masks):
        pass


class QwenReward(base_value_model):
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
        
        self.system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
            
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
    
    def get_value(self, query, output):
        try:
            formatted_steps = self.format_steps(output)
            input_ids, token_masks = self.prepare_input(query, formatted_steps)
            step_rewards = self.compute_rewards(input_ids, token_masks)
            return step_rewards[0][-1] # last step reward is better
        except Exception as e:
            print(f"Error in get_value: {str(e)}")
            return self.low