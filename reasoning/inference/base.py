import os
import json
import numpy as np
class BaseInference:
    # This is the base class for all inference methods. It contains the common methods for all inference methods.
    def __init__(self, policy_model, tokenizer, sampling_params, config_name, reward_model, method, ORM_type):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.config_name = config_name
        self.reward_model = reward_model
        self.method = method
        self.ORM_type = ORM_type
        # metrics
        self.sample_size = 0
        self.right_count = 0
        self.num_generated_tokens = [] # The total number of generated tokens for each question
        self.num_input_tokens = [] # The total number of input tokens for each question

        # methods
        self.method_dict = {
            'majority': self.majority_vote,
            'weighted_majority': self.weighted_majority_vote,
            'best_of_N': self.best_of_N,
            'beam_search': self.beam_search,
       }
        # This is a jsonl file that contains the results of the inference. If already exists, it will be overwritten.
        results_dir = "/ssdscratch/byuan48/efficient_reasoning/results"
        os.makedirs(results_dir, exist_ok=True)
        self.results_file = os.path.join(results_dir, f"{self.config_name}.jsonl")
        if os.path.exists(self.results_file): 
            os.remove(self.results_file)

    def reset(self):
        self.sample_size = 0
        self.right_count = 0
        self.num_generated_tokens = []
        self.num_input_tokens = []
    
    def get_reward(self, rewards):
        if self.ORM_type == 'min':
            return min(rewards)
        elif self.ORM_type == 'last':
            return rewards[-1]
        elif self.ORM_type == 'product':
            return np.prod(rewards)
        elif self.ORM_type == 'geo_mean':
            return np.exp(np.mean(np.log(rewards)))
        else:
            raise ValueError(f"Invalid ORM type: {self.ORM_type}")
    def generate_text(self, system_prompt, questions): # Generate text using the chat template
        sampling_params = self.sampling_params
        conversations = [[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ] for question in questions]

        outputs = self.policy_model.chat(conversations, sampling_params)
        texts = []
        num_generated_tokens  = [] 
        num_input_tokens = [] # Only count once for batch inference. This is to match the lower bound
        for output in outputs:
            for gen_output in output.outputs:
                texts.append(gen_output.text)
                num_generated_tokens.append(len(gen_output.token_ids))
            num_input_tokens.append(len(output.prompt_token_ids))
        # print(texts)

        # assert len(num_generated_tokens ) == len(num_input_tokens) * sampling_params.n

        return texts, num_generated_tokens, num_input_tokens
    
    def generate_text_completion(self, system_prompt, questions, partial_solutions): # Generate text using the chat template
        sampling_params = self.sampling_params
        conversations = [[
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
                "content": partial_solution + '\n\n'
            } 
        ] for query, partial_solution in zip(questions, partial_solutions)]

        prompt_token_ids = [self.tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in conversations]
        # remove the last 5 tokens for chat completion. Need test new models to see if this default number is right
        # Add special token '\n\n' to the end of the prompt
        token_id_to_add = self.tokenizer.encode('\n\n')[-1]
        prompt_token_ids = [prompt_token_id[:-5] + [token_id_to_add] for prompt_token_id in prompt_token_ids] 
        outputs = self.policy_model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)


        # outputs = self.policy_model.generate(conversations, sampling_params)
        texts = []
        num_generated_tokens  = [] 
        num_input_tokens = [] # Only count once for batch inference. This is to match the lower bound
        for output in outputs:
            for gen_output in output.outputs:
                texts.append(gen_output.text)
                num_generated_tokens.append(len(gen_output.token_ids))
            num_input_tokens.append(len(output.prompt_token_ids))
        # print(texts)

        # assert len(num_generated_tokens ) == len(num_input_tokens) * sampling_params.n

        return texts, num_generated_tokens, num_input_tokens

    def save_solutions_to_jsonl(self, data):
        # data is a dictionary or a list of dictionaries
        with open(self.results_file, 'a') as f:
            if isinstance(data, list):
                # If data is a list of dictionaries, write each one as a JSON line
                for item in data:
                    f.write(json.dumps(item) + '\n')
            else:
                # If data is a single dictionary, write it as a JSON line
                f.write(json.dumps(data) + '\n')

    def majority_vote(self, generated_text, rewards):
        pass

    def weighted_majority_vote(self, generated_text, rewards):
        pass

    def best_of_N(self, generated_text, rewards):
        pass

    def beam_search(self, generated_text, rewards):
        pass