import torch
from reasoning.evaluator.math_grader import math_equal, extract_answer
from vllm import SamplingParams
# to do: multiple gpus
class GreedyInference:
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        torch.backends.cudnn.benchmark = True  # 启用 cuDNN benchmark 模式
        self.right_count = 0
        self.sample_size = 0
        
    def reset(self):
        self.right_count = 0
        self.sample_size = 0
        
    def generate_text(self, input_text, max_new_tokens=1024):
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            n=1,    
            stop_token_ids=[self.tokenizer.eos_token_id],
            skip_special_tokens = True,
            include_stop_str_in_output = False,
        )
        outputs = self.model.generate(input_text, sampling_params)
        texts = []
        for output in outputs:
            for gen_output in output.outputs:
                texts.append(gen_output.text)
        return texts
    
    def inference(self, questions, answers):
        generated_solutions = self.generate_text(questions)
        self.sample_size += len(questions)
        for i in range(len(questions)):
            selected_solution = extract_answer(generated_solutions[i])
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
    print("hello. This is greedy inference")