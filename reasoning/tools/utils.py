# import re
# from sympy import sympify, simplify, N
# from sympy.parsing.latex import parse_latex
import json, yaml
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from vllm import LLM   
import numpy as np
import random
from transformers import set_seed

# def extract_boxed_answers(solutions_batch):
#     # print('solutions_batch:',solutions_batch)
#     """Extracts all answers inside \boxed{} for a batch of generated solutions."""
#     all_results = []
#     num_invalid_solutions = 0
#     indices_invalid = []
#     # Regex to match \boxed{...} with potential nested braces
#     regex = r"\\boxed\{((?:[^{}]|(?:\{(?:[^{}]|(?:\{.*?\}))*\}))*)\}"
#     for i, solutions in enumerate(solutions_batch):
#         results = []
#         for j, solution_text in enumerate(solutions):
#             # Find all matches for \boxed{}
#             matches = re.findall(regex, solution_text)
#             if matches and len(set(matches)) == 1:
#                 results.append(matches[0]) # Only one match found
#             elif matches and len(set(matches)) > 1:
#                 # print(f"Warning: Multiple matches found in solution: {matches} at index {solutions.index(solution_text)}")
#                 # print(f'solution_text:',solution_text)
#                 num_invalid_solutions += 1
#                 indices_invalid.append((i,j))
#                 results.append(matches[-1]) # use the last match
#             else:
#                 results.append(None)
#                 # print(f"Warning: No match found in solution: {solution_text} at index {solutions.index(solution_text)}")
#                 num_invalid_solutions += 1
#                 indices_invalid.append((i,j))
#         all_results.append(results)
#     print('all_results:',all_results)
#     return all_results, num_invalid_solutions, indices_invalid

# def is_numeric(expr):
#     """Checks if the expression can be evaluated to a numeric result."""
#     try:
#         N(expr)  # Try numerical evaluation
#         return True
#     except:
#         return False    

# # def check_equivalence(expressions, answer,tolerance=1e-3):
# #     """Checks if a list of expressions are equivalent."""
# #     if answer is None:
# #         return False, [False] * len(expressions) 
# #     parsed_expressions = []
# #     for expr in expressions:
# #         if expr is not None:
# #             parsed_expr = expr
# #             try:
# #                 # Attempt LaTeX parsing
# #                 parsed_expr = parse_latex(expr)
# #             except Exception:
# #                 try:
# #                     # Fallback to plain mathematical parsing
# #                     if '/' in expr:
# #                         parsed_expr = Rational(expr)
# #                     else:
# #                         parsed_expr = sympify(expr)
# #                 except Exception as e:
# #                     print(f"Error parsing expression '{expr}': {e}; using the original text: {expr}")
# #             parsed_expressions.append(parsed_expr)
# #         else:
# #             parsed_expressions.append(None)
    
# #     # print('parsed_expressions:',parsed_expressions)
# #     # Check if any expressions can be evaluated as numbers
# #     if is_numeric(answer):
# #         # Perform numerical equivalence check
# #         numerical_values = [N(expr)  for expr in parsed_expressions if expr is not None else None]
# #         reference_value = N(answer)
# #         # print(numerical_values)
# #         comparison = [abs(reference_value - value) < tolerance for value in numerical_values if is_numeric(value)] 
# #         return any(comparison), comparison
# #     else:
# #         # Fall back to symbolic equivalence check
# #         simplified_forms = [simplify(expr) for expr in parsed_expressions if expr is not None else None]
# #         reference_form = simplify(parse_latex(answer))
# #         # print(simplified_forms)
# #         comparison = [reference_form == form for form in simplified_forms]
# #         return any(comparison), comparison

# from sympy import simplify, sympify, N, Rational
# from sympy.parsing.latex import parse_latex

# def check_equivalence(expressions, answer, tolerance=1e-3):
    
#     # special case: ^\circ is in the answer; \text{answer}
#     answer = answer.replace(r'^\circ', '')

#     if answer is None:
#         return False, [False] * len(expressions)
    
#     # Parse the answer
#     try:
#         reference_value = N(answer) if is_numeric(answer) else simplify(parse_latex(answer))
#     except Exception as e:
#         # print(f"Error parsing answer '{answer}': {e}")
#         return False, [False] * len(expressions)

#     parsed_expressions = []
#     for expr in expressions:
#         if expr is not None:
#             try:
#                 # Attempt LaTeX parsing
#                 parsed_expr = parse_latex(expr)
#             except Exception:
#                 try:
#                     # Fallback to plain mathematical parsing
#                     parsed_expr = Rational(expr) if '/' in expr else sympify(expr)
#                 except Exception as e:
#                     # print(f"Error parsing expression '{expr}': {e}; using the original text.")
#                     parsed_expr = None
#             parsed_expressions.append(parsed_expr)
#         else:
#             parsed_expressions.append(None)
    
#     # Compare parsed expressions to the answer
#     comparisons = []
#     for parsed_expr in parsed_expressions:
#         if parsed_expr is None:
#             comparisons.append(False)
#         else:
#             if is_numeric(answer):
#                 try:
#                     value = N(parsed_expr)
#                     comparisons.append(abs(reference_value - value) < tolerance)
#                 except Exception:
#                     comparisons.append(False)
#             else:
#                 try:
#                     comparisons.append(simplify(parsed_expr) == reference_value)
#                 except Exception:
#                     comparisons.append(False)
#     # print(comparisons)
#     for index, comparison in enumerate(comparisons):
#         try:
#             re = bool(comparison)
#         except Exception:
#             re = False
#         comparisons[index] = re
#     # print(comparisons)
#     is_any_correct = any(comparisons)   
#     return is_any_correct, comparisons




# def save_batch_solutions_to_jsonl(batch_solutions, input_texts, output_file):
#     """Saves a batch of generated solutions to a JSONL file."""
#     with jsonlines.open(output_file, mode="a") as writer:  # Append mode
#         for input_text, solutions in zip(input_texts, batch_solutions):
#             json_entry = {
#                 "input": input_text,
#                 "solutions": solutions  # List of generations for this input
#             }
#             writer.write(json_entry)

# def return_entropy(logprobs):
#     entropies = []
#     num_tokens = len(logprobs) 
#     for i in range(num_tokens):
#         temp = []
#         for token_id in logprobs[i].keys():
#             temp.append(logprobs[i][token_id].logprob)
#         temp = np.array(temp)
#         entropy = -1 * np.sum(np.exp(temp) * temp)
#         entropies.append(entropy)
#     return entropies

# def return_probs(logprobs):
#     probs = []
#     num_tokens = len(logprobs)
#     for i in range(num_tokens):
#         token_id = next(iter(logprobs[i]))
#         probs.append(logprobs[i][token_id].logprob)
#     return probs


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model(model_name, device="cuda:0"):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype = torch.bfloat16, 
        trust_remote_code = True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = 'left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def load_model_with_vllm(model_name, tensor_parallel_size = 2):

    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)