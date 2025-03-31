import requests
import json

class Base_API_Model:
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.api_key = api_key if api_key else "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"
        
    def get_response(self, prompt):
        pass


class OpenRouter_API_Model(Base_API_Model):
    def __init__(self, model_name, api_key=None, temperature=1, top_p=0.9, max_tokens=256, seed=40):
        super().__init__(model_name, api_key)

        self.model_name = "openai/o1-mini-2024-09-12"
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        
    def get_response(self, prompt):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "logprobs": True,  # Request logprobs in the response
                "top_logprobs": 5,  # Get probabilities for top 5 tokens at each position
                # Generation parameters
                "temperature": self.temperature,  # Controls randomness (0.0 to 2.0)
                "top_p": self.top_p,            # Nucleus sampling parameter
                "max_tokens": self.max_tokens,   # Maximum length of response
                "seed": self.seed,              # Random seed for reproducibility
                "stream": False            # Whether to stream the response
            })
        )
        
        response_json = response.json()
        return {
            'content': response_json['choices'][0]['message']['content'],
            'usage': response_json['usage'],
            'logprobs': response_json['choices'][0].get('logprobs', {})  # Get logprobs if available
        }
    