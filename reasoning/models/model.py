# This file contains Qwen and Llama models loaded via transformers
from reasoning.tools.utils import load_model
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch


class BasePolicyModel:
    def __init__(self, model_name, device="cuda:0"):
        self.model, self.tokenizer = load_model(model_name, device)
        self.device = device
    
    def get_proposal(self, prompt):
        pass

    def test(self):
        # test the model with a simple query to see if the format is correct
        query = r"""Given a science problem and an existing incomplete solution, your task is to complete the solution in a smooth and proper way.

            - If no existing steps are provided, you must briefly analyze the problem and output only the first step.  
            - If existing steps are sufficent to solve the problem, you must output the final answer and format the answer inside `\\boxed{}` (e.g., `\\boxed{42}`). 
            - If existing steps are not sufficent to solve the problem, you must output exactly **one** correct next step that naturally follows from the previous ones.  
            - You **must** follow the given format.

            **Strict Output Format:**  
            - Your response must always start with: `Next step: ...`  
            - The response must be limited to one reasoning step (e.g., a calculation, reasoning, or answer choice).  


            If there are multiple reasonable next steps, choose the most natural one based on the provided existing steps.  

            Here is the problem and the existing steps:  

            Problem: How many positive whole-number divisors does 196 have?
            Existing Steps:
            Step 1: Factorize 196 into its prime factors to determine its divisors. To find the number of positive whole-number divisors of 196, we first need to factorize 196 into its prime factors.  ##
            Step 2: The prime factorization of 196 is $2^2 \cdot 7^2$.
            Step 3: Understand the prime factorization of 196 The prime factorization of 196 is given as $2^2 \cdot 7^2$. This tells us that the number 196 has two distinct prime factors, 2 and 7, with 2 raised to the power of 2 and 7 raised to the power of 2.  ##

            Output:"""
        response = self.get_proposal(query)
        print(f'response: {response}')

class LlamaPolicyModel(BasePolicyModel): # model should be used for all Llama, Qwen and Qwen-Distall-R1 models
    def __init__(self, model_name, device="cuda:0", 
                 temperature=0.7, top_p=0.9, num_return_sequences=1, 
                 max_new_tokens=128, do_sample=True, max_tokens=None):
        super().__init__(model_name, device)
        self.temperature = temperature
        self.top_p = top_p
        self.num_return_sequences = num_return_sequences
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.max_tokens = max_tokens
    
    def get_proposal(self, prompt):
        return self.get_local_response_llama(prompt)
        
    def get_local_response_llama(self,query):
        cnt = 2
        all_response = ''
        # messages = [{"role": "user", "content": query}]
        # data = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        message = '<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'.format(query=query)
        data = self.tokenizer.encode_plus(message, return_tensors='pt')
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        while cnt:
            try:
                # print(f'input_ids: {input_ids}')
                # print(f'attention_mask: {attention_mask}')
                output = self.model.generate(input_ids, attention_mask=attention_mask, do_sample=self.do_sample, max_new_tokens=self.max_new_tokens, temperature=self.temperature, eos_token_id=terminators, pad_token_id=self.tokenizer.eos_token_id)
                ori_string = self.tokenizer.decode(output[0], skip_special_tokens=False)
                print(f'ori_string: {ori_string}')
                processed_string = ori_string.split('<|end_header_id|>')[2].strip().split('<|eot_id|>')[0].strip()
                response = processed_string.split('<|end_of_text|>')[0].strip()
                all_response = response
                break
            except Exception as e:
                print(f'Error:{e}, obtain response again...\n')
                cnt -= 1
        if not cnt:
            return []
        # split_response = all_response.split("Assistant:")[-1].strip().split('\n')
        split_response = all_response.split('\n')
        return split_response
    

class QwenPolicyModel(BasePolicyModel): # model should be used for all Llama, Qwen and Qwen-Distall-R1 models
    def __init__(self, model_name, device="cuda:0", 
                 temperature=0.7, top_p=0.9, num_return_sequences=1, 
                 max_new_tokens=128, do_sample=True, max_tokens=None):
        super().__init__(model_name, device)
        self.temperature = temperature
        self.top_p = top_p
        self.num_return_sequences = num_return_sequences
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.max_tokens = max_tokens
    
    def get_proposal(self, prompt):
        return self.get_local_response_qwen(prompt)
        
    def get_local_response_qwen(self,query):
        cnt = 2
        all_response = ''
        # messages = [{"role": "user", "content": query}]
        # data = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        message = '<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'.format(query=query)
        data = self.tokenizer.encode_plus(message, return_tensors='pt')
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        while cnt:
            try:
                print(f'input_ids: {input_ids}')
                print(f'attention_mask: {attention_mask}')
                output = self.model.generate(input_ids, attention_mask=attention_mask, do_sample=self.do_sample, max_new_tokens=self.max_new_tokens, temperature=self.temperature, eos_token_id=terminators, pad_token_id=self.tokenizer.eos_token_id)
                ori_string = self.tokenizer.decode(output[0], skip_special_tokens=False)
                print(f'ori_string: {ori_string}')
                processed_string = ori_string.split('<|end_header_id|>')[2].strip().split('<|eot_id|>')[0].strip()
                response = processed_string.split('<|end_of_text|>')[0].strip()
                all_response = response
                break
            except Exception as e:
                print(f'Error:{e}, obtain response again...\n')
                cnt -= 1
        if not cnt:
            return []
        # split_response = all_response.split("Assistant:")[-1].strip().split('\n')
        split_response = all_response.split('\n')
        return split_response



class ValueModel_shepherd: # not very good use the next one instead
    def __init__(self, device, low = 0, good_token='+', bad_token='-', step_tag='ки'):

        self.model_name = 'peiyi9979/math-shepherd-mistral-7b-prm'
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).eval().to(self.device)
        
        
        self.good_token = good_token
        self.bad_token = bad_token
        self.step_tag = step_tag
        self.candidate_tokens = self.tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
        self.step_tag_id = self.tokenizer.encode(step_tag)[-1]
        self.low = low
        
    def format_steps(self, input_text):

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
            
            step = step.replace(f"Step {i}:", f"Step {i}:")
            
            step = step.replace('##', '').strip()
            
            formatted_step = f"{step} ки"
            formatted_steps.append(formatted_step)
            
        return '\n'.join(formatted_steps)
            

    def get_value(self, question, output): 

        try:
            formatted_steps = self.format_steps(output)
            input_text = f"{question} {formatted_steps}"  
            input_ids = torch.tensor([self.tokenizer.encode(input_text)]).to(self.device)

            with torch.no_grad():
                
                logits = self.model(input_ids).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[:, :, 0]  
                
                step_mask = (input_ids == self.step_tag_id).cpu()
                step_scores = scores[step_mask].tolist()

            if not step_scores:
                return self.low

            
            last_step_score = step_scores[-1]
            return last_step_score

        except Exception as e:
            print(f"wrong evaluation: {str(e)}")
            return self.low


class ValueModel_qwen:
    def __init__(self, device, low = 0):
        self.device = device
        self.model_name = "Qwen/Qwen2.5-Math-PRM-7B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        self.system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        self.low = low
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
                
    def get_value(self, query, output): # used in the code of MCTS
        try:
            formatted_steps = self.format_steps(output)
            input_ids, token_masks = self.prepare_input(query, formatted_steps)
            step_rewards = self.compute_rewards(input_ids, token_masks)
            return step_rewards[0][-1] # last step reward is better
        except Exception as e:
            print(f"Error in get_value: {str(e)}")
            return self.low

    def get_value_with_steps(self, query, steps): # used in the code of SamplingTree
        try:
            input_ids, token_masks = self.prepare_input(query, steps)
            step_rewards = self.compute_rewards(input_ids, token_masks)
            return step_rewards[0]
        except Exception as e:
            print(f"Error in get_value: {str(e)}")
            return self.low