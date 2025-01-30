from reasoning.tools.utils import load_model
import random
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
    
    def get_value(self, prompt): # to implement
        return random.random() / 2
    
    
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