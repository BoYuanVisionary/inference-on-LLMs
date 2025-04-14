from reasoning.inference.base import BaseInference
from reasoning.tools.utils import load_model_with_vllm
from reasoning.evaluator.math_grader import math_equal, extract_answer
from datasets import load_dataset
from vllm import SamplingParams
import wandb
import time
from reasoning.tools.logger import load_config, apply_config
import numpy as np
import argparse


class BeamInference(BaseInference):
    def __init__(self, policy_model, tokenizer, sampling_params, config_name, reward_model, max_steps, beam_width):
        super().__init__(policy_model, tokenizer, sampling_params, config_name, reward_model, method='beam_search')   

        self.next_step_prompt = '''
        Given a math problem and an existing incomplete solution, your task is to complete the solution in a smooth and proper way.

        - If no existing steps are provided, you must briefly analyze the problem and output only the first step.  
        - If existing steps are sufficent to solve the problem, you must output the final answer and format the answer inside `\\boxed{}` (e.g., `\\boxed{42}`). 
        - If existing steps are not sufficent to solve the problem, you must output exactly **one** correct next step that naturally follows from the previous ones.  
        - You **must** follow the given format.

        **Strict Output Format:**  
        - Your response must always start with: `Next step: ...`  
        - The response must be limited to one reasoning step (e.g., a calculation, reasoning, or answer choice).  


        If there are multiple reasonable next steps, choose the most natural one based on the provided existing steps.  

    '''
        self.max_steps = max_steps
        self.beam_width = beam_width
        self.sampling_params.n = beam_width # fixed beam width

    def get_next_step(self, x, y, step_n):

        if y == '':
            y = 'None\n'
        question = "Here is the problem and the existing steps:\n Problem: "+ x + '\nExisting Steps:\n' + y + '\nOutput:'
        responses, num_generated_tokens, num_input_tokens = self.generate_text(self.next_step_prompt, [question])
        output = []
        for response in responses:
            if y == 'None\n': # only count the input tokens once in the first step
                self.num_input_tokens.append(num_input_tokens) 

            if not response:
                print('next step is empty！\n')
                output.append({'next_step': '', 'reward': 0, 'token_length': 0})
                continue


            p = response.strip()

            if "Next step:" in p:
                stp = p.split('Next step:')[1].strip()
                if len(stp) < 2:
                    print('next step is too short！\n')
                    output.append({'next_step': '', 'reward': 0, 'token_length': 0})
                    continue
                if stp in y:
                    print('next step is repeated！\n')
                    output.append({'next_step': '', 'reward': 0, 'token_length': 0})
                    continue

                revised_ = 'Step ' + str(step_n) + ': ' + stp
                print(f'standardized next step: {revised_}\n')
                output.append({'next_step': revised_ + '\n', 'reward': 0, 'token_length': num_generated_tokens})

            elif "Step" in p and ":" in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('Step')[0].strip()
                if len(p_) < 4:
                    print('next step is too short！\n')
                    output.append({'next_step': '', 'reward': 0, 'token_length': 0})
                    continue
                p_ = p_[1:].strip()
                if p_ in y:
                    print('next step is repeated！\n')
                    output.append({'next_step': '', 'reward': 0, 'token_length': 0})
                    continue

                revised_ = 'Step ' + str(step_n) + ': ' + p_
                print(f'standardized next step: {revised_}\n')
                output.append({'next_step': revised_ + '\n', 'reward': 0, 'token_length': num_generated_tokens})

            else:
                p_ = p.strip()
                if len(p_) < 3:
                    print('next step is too short！\n')
                    output.append({'next_step': '', 'reward': 0, 'token_length': 0})
                    continue
                if p_ in y:
                    print('next step is repeated！\n')
                    output.append({'next_step': '', 'reward': 0, 'token_length': 0})
                    continue

                revised_ = 'Step ' + str(step_n) + ': ' + p_
                print(f'standardized next step: {revised_}\n')
                output.append({'next_step': revised_ + '\n', 'reward': 0, 'token_length': num_generated_tokens})

        return output

    def inference(self, question, answer): # system_prompt is self.next_step_prompt

        self.sample_size += 1

        # no batch inference for beam search
        solutions = []
            
        remaining_steps = self.max_steps
        exsiting_steps = ''
        step_n = 1
        generated_tokens_sample = 0
        while remaining_steps > 0:
            output = self.get_next_step(question, exsiting_steps, step_n) # output is a list of dicts where each dict contains 'next_step' and 'reward'
            generated_tokens_sample += np.sum([item['token_length'] for item in output])
            remaining_steps -= 1
            step_n += 1
            feasible_steps = []
            for item in output:
                if item['next_step'] != '':
                    feasible_steps.append(item['next_step'])
                    reward = self.reward_model.get_value_with_steps(question+'\n'+exsiting_steps, [item['next_step']])
                    item['reward'] = reward[0]
            if len(feasible_steps) == 0:
                break
            else:
                max_reward_idx = max(range(len(output)), key=lambda i: output[i]['reward'])
                next_step = feasible_steps[max_reward_idx]
                assert next_step != ''
                exsiting_steps = exsiting_steps +'\n'+ next_step
                if extract_answer(next_step) is not None:
                    break
        solutions.append(exsiting_steps)

        extracted_answer = extract_answer(answer)
        selected_solution = extract_answer(exsiting_steps)
        is_correct = math_equal(selected_solution, extracted_answer)
        if extracted_answer is None:
            raise ValueError('extracted answer is None')
        if is_correct:
            self.right_count += 1
        
        print(f'question: {question}')
        print(f'solution: {selected_solution}')
        print(f'extracted answer: {extracted_answer}')
        print(f'is correct: {is_correct}')
        print(f'num_generated_tokens: {(generated_tokens_sample)}')
        print(f'final_steps: {step_n-1}')
        print("--------------------------------")

        save_data = {
            "question": question,
            "selected_solution": selected_solution,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "num_generated_tokens": generated_tokens_sample,
            "final_steps": step_n-1,
        }
        self.save_solutions_to_jsonl(save_data)
        print(f'processed {self.sample_size} samples')

        return self.right_count / self.sample_size

        
if __name__ == "__main__":
    
    # use parser to parse the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../../configs/development.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    apply_config(config) # set up wandb, seed and cuda device
    print(config)

    # load dataset and model
    dataset = load_dataset(config["data_path"])
    dataset = dataset['test']
    model_name = config["model_name"]
    model, tokenizer = load_model_with_vllm(model_name, task='auto', tensor_parallel_size=len(config["cuda_device_ids"]), gpu_memory_utilization=0.9)
    tokenizer.pad_token = tokenizer.eos_token 
    reward_model = config.get("reward_model", None)

    # inference hyperparameters
    num_return_sequences = config["num_return_sequences"]
    max_new_tokens = config["max_new_tokens"]
    temperature = config["temperature"]
    top_p = config["top_p"]
    batch_size = config["batch_size"]
    num_return_sequences = config["num_return_sequences"]
    system_prompt = config["system_prompt"]

    # name of the results file
    config_name = config["config_name"]

    # Also set the seed for the sampling params 
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=num_return_sequences,
        top_p=top_p,
        stop_token_ids=[tokenizer.eos_token_id],
        skip_special_tokens = True,
        include_stop_str_in_output = False,
    ) # shouldn't set seed for random sampling

    inference = BeamInference(model, tokenizer,sampling_params,config_name,reward_model,method='beam_search')

    start_time = time.time()
    for i in range(0, len(dataset)):
        question = dataset['problem'][i]
        answer = dataset['solution'][i]            
        accuracy = inference.inference(system_prompt, question, answer)
        print("Accuracy: {}".format(accuracy))
        inference.reset()
    wandb.log({"accuracy": accuracy})
    end_time = time.time()
    print(f"parallel size: {num_return_sequences}")
    print(f"generated tokens per sample in average: {np.mean(inference.num_generated_tokens) * num_return_sequences}")
    wandb.log({"generated tokens per sample in average": np.mean(inference.num_generated_tokens) * num_return_sequences})
    print("Time taken: {} seconds".format(end_time - start_time))

    wandb.finish()