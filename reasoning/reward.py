import torch

class Reward:
    
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer


    def score_with_openai(self, text):
        raise NotImplementedError('score_with_openai is not implemented')


    def score_with_log_prob(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        avg_log_prob = log_probs.mean().item()
        # Scale the log probability to a score between 1 to 5
        score = (avg_log_prob + 10) / 2
        return score

    def get_reward(self, text, method="llm"):
        if method == "llm":
            return self.score_with_llm(text)
        elif method == "log_prob":
            return self.score_with_log_prob(text)
        else:
            raise ValueError("Invalid method. Choose either 'llm' or 'log_prob'.")
