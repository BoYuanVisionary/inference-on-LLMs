import random
from tasks.science import SearchTask
from MCTS.base import treeNode
from models.get_response import *
from MCTS.mcts import MCTS
from reasoning.evaluator.math_grader import math_equal, extract_answer


class MCTS_Task(SearchTask):
    def __init__(self, data, propose_method='glm', value_method='glm', branch=3, end_gate=0.9, roll_policy='greedy',
                 roll_branch=1, roll_forward_steps=3, time_limit=None, iteration_limit=None, exploration_constant=0.7,
                 alpha=0.5, inf=1.0, temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, low=0, high=1,
                 evaluate='', sample_value='simple', answer=None, verify_method='string', weighted_verify=False):
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.mode = 'mcts'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.branch = branch
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.end_gate = end_gate
        self.roll_policy = roll_policy
        self.roll_branch = roll_branch
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.roll_forward_steps = roll_forward_steps
        self.alpha = alpha
        self.limit_type = None
        self.INF = inf
        self.node_count = 1
        self.sample_value = sample_value
        self.answer = answer
        self.verify_method = verify_method
        self.weighted_verify = weighted_verify

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1

    def set_limit_type(self):
        if self.time_limit is not None:
            if self.iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.limit_type = 'time'
        else:
            if self.iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if self.iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.limit_type = 'iterations'

    def get_next_step(self, y, step_n):
        if self.use_case_prompt:
            prompt = self.single_propose_prompt_wrap(self.question, y, step_n) # this is to generate the next step
        else:
            if self.propose_method == 'gpt':
                prompt = self.zero_single_propose_wrap_gpt(self.question, y, step_n, self.lang)
            elif self.propose_method == 'mistral' or self.propose_method == 'llama':
                prompt = self.zero_single_propose_wrap_mistral(self.question, y, step_n)
            else:
                prompt = self.zero_single_propose_wrap(self.question, y, step_n, self.lang)

        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            print('获得下一步失败！\n')
            return ''

        if len(response) > 5:
            response = response[:5]

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        
        if "Next step:" in p:
            stp = p.split('Next step:')[1].strip()
            if len(stp) < 2:
                print('输出步骤过短！\n')
                return ''
            if stp in y:
                print('输出步骤重复！\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + stp
            print(f'标准化后新的步骤:{revised_}\n')
            return revised_ + '\n'

        elif "Step" in p and ":" in p:
            pre_len = len(p.split(':')[0])
            p_ = p[pre_len:]
            p_ = p_.split('Step')[0].strip()
            if len(p_) < 4:
                print('输出步骤过短！\n')
                return ''
            p_ = p_[1:].strip()
            if p_ in y:
                print('输出步骤重复！\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + p_
            print(f'标准化后新的步骤:{revised_}\n')
            return revised_ + '\n'

        else:
            p_ = p.strip()
            if len(p_) < 3:
                print('输出步骤过短！\n')
                return ''
            if p_ in y:
                print('输出步骤重复！\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + p_
            print(f'标准化后新的步骤:{revised_}\n')
            return revised_ + '\n'


    def get_step_value(self, y):
        if y in self.value_cache.keys():
            return self.value_cache[y]

        if self.value_method == 'local':
            if self.lang == 'zh':
                prompt_answer = '问题:' + self.question + '\n步骤:\n' + '【答案】' + y
            else:
                prompt_answer = 'Problem: ' + self.question + '\nSolution:\n' + y
            value = get_value(prompt_answer, self.value_method, self.temperature, self.max_tokens, self.seed,
                              self.max_length, self.low, self.high)
            print(f'获得评分:{value}\n')
            self.value_cache.update({y: value})
            return value

        else:
            prompt = self.value_prompt_wrap(self.question, y)
            response = get_value(prompt, self.value_method, self.temperature, self.max_tokens, self.seed,
                                 self.max_length, self.low, self.high)
            value = self.value_outputs_unwrap(response, self.low, self.high)
            print(f'获得评分:{value}\n')
            self.value_cache.update({y: value})
            return value



    def run(self):
        self.clear_cache()
        self.set_limit_type()
        node, finish, root = MCTS(self)
 
        solution = node.y
        
        extracted_answer = extract_answer(solution)
        ground_truth = extract_answer(self.answer)
        if ground_truth is None:
            raise ValueError('ground_truth is None')
        correctness =  math_equal(extracted_answer, ground_truth)

        final_answer = {'content': self.question, 'extracted_answer': extracted_answer,  'finish': finish,
                        'real_answer': self.ground_truth, 'correctness': correctness}
        return final_answer, root
           