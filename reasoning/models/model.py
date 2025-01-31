from reasoning.tools.utils import load_model
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Model:
    def __init__(self, model_name, device="cuda:0", 
                 temperature=0.7, top_p=0.9, num_return_sequences=1, 
                 max_new_tokens=128, do_sample=True, max_tokens=None):
        self.model, self.tokenizer = load_model(model_name, device)
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.num_return_sequences = num_return_sequences
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.max_tokens = max_tokens
    
    def update_generation_settings(self, **kwargs):
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.9)
        self.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 128)
        self.do_sample = kwargs.get("do_sample", True)
        self.max_tokens = kwargs.get("max_tokens", 1024)

    # def generate(self, prompt):
    #     print(prompt)
    #     inputs = self.tokenizer(prompt, return_tensors="pt",padding=True,truncation=True).to(self.device)
    #     outputs = self.model.generate(
    #         **inputs,
    #         temperature=self.temperature,
    #         top_p=self.top_p,
    #         num_return_sequences=self.num_return_sequences,
    #         max_new_tokens=self.max_new_tokens,
    #         do_sample=self.do_sample,
    #         max_tokens=self.max_tokens,
    #         pad_token_id=self.tokenizer.eos_token_id
    #         )
    #     generation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    #     print(generation[len(prompt)-1:])
    #     return generation[len(prompt)-1:]
    
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
                output = self.model.generate(input_ids, attention_mask=attention_mask, do_sample=self.do_sample, max_new_tokens=self.max_new_tokens, temperature=self.temperature, eos_token_id=terminators, pad_token_id=self.tokenizer.eos_token_id)
                ori_string = self.tokenizer.decode(output[0], skip_special_tokens=False)
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
    


class ValueModel_shepherd:
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
            

    def get_value(self, question, output): # this is to return the mean reward of all steps

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

            avg_score = sum(step_scores) / len(step_scores)
            last_step_score = step_scores[-1]
            # print(step_scores)
            return avg_score

        except Exception as e:
            print(f"wrong evaluation: {str(e)}")
            return self.low
