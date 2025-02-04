import random
from reasoning.tasks.science import SearchTask
from reasoning.VMCTS.base import treeNode
from reasoning.VMCTS.mcts import MCTS
from reasoning.evaluator.math_grader import math_equal, extract_answer
import time

class VMCTS_Task(SearchTask):
    def __init__(self, data, propose_method=None, value_method=None, branch=3, end_gate=0.9, roll_policy='greedy',
                 roll_branch=1, roll_forward_steps=2, time_limit=None, iteration_limit=None, exploration_constant=0.7,
                 alpha=0.5, inf=1.0, temperature=0.7, max_tokens=2048, seed=110, max_length=2048,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, low=0, high=1,
                 evaluate='', sample_value='simple', answer=None, verify_method='string', weighted_verify=False, root_variance=0.25, gamma = 10):
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.mode = 'mcts'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
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
        self.root_variance = root_variance
        self.gamma = gamma
        assert propose_method is not None, "propose_method is required"
        assert value_method is not None, "value_method is required"
        
        

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
        # if self.use_case_prompt:
        #     prompt = self.single_propose_prompt_wrap(self.question, y, step_n) # this is to generate the next step
        # else:
        #     if self.propose_method == 'gpt':
        #         prompt = self.zero_single_propose_wrap_gpt(self.question, y, step_n, self.lang)
        #     elif self.propose_method == 'mistral' or self.propose_method == 'llama':
        #         prompt = self.zero_single_propose_wrap_mistral(self.question, y, step_n)
        #     else:
        #         prompt = self.zero_single_propose_wrap(self.question, y, step_n, self.lang)
        prompt = self.zero_single_propose_wrap_mistral(self.question, y, step_n)

        response = self.propose_method.get_proposal(prompt)
        if not response:
            print('next step is empty！\n')
            return ''

        if len(response) > 5: # if there are more than 5 setences splited by '\n'
            response = response[:5]

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        
        if "Next step:" in p:
            stp = p.split('Next step:')[1].strip()
            if len(stp) < 2:
                print('next step is too short！\n')
                return ''
            if stp in y:
                print('next step is repeated！\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + stp
            print(f'standardized next step: {revised_}\n')
            return revised_ + '\n'

        elif "Step" in p and ":" in p:
            pre_len = len(p.split(':')[0])
            p_ = p[pre_len:]
            p_ = p_.split('Step')[0].strip()
            if len(p_) < 4:
                print('next step is too short！\n')
                return ''
            p_ = p_[1:].strip()
            if p_ in y:
                print('next step is repeated！\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + p_
            print(f'standardized next step: {revised_}\n')
            return revised_ + '\n'

        else:
            p_ = p.strip()
            if len(p_) < 3:
                print('next step is too short！\n')
                return ''
            if p_ in y:
                print('next step is repeated！\n')
                return ''

            revised_ = 'Step ' + str(step_n) + ': ' + p_
            print(f'standardized next step: {revised_}\n')
            return revised_ + '\n'


    def get_step_value(self, y):
        if y in self.value_cache.keys():
            return self.value_cache[y]
        # print('Problem: ' + self.question + '\nSolution:\n' + y)
        value = self.value_method.get_value(self.question, y)
        print(f'获得评分:{value}\n')
        self.value_cache.update({y: value})
        return value

    def run(self):
        self.clear_cache()
        self.set_limit_type()
        time_start = time.time()
        node, finish, root = MCTS(self)
        time_end = time.time()
        solution = node.y
        print(f'solution: {solution}\n')
        extracted_answer = extract_answer(solution)
        ground_truth = extract_answer(self.answer)
        if ground_truth is None:
            raise ValueError('ground_truth is None')
        correctness =  math_equal(extracted_answer, ground_truth)

        final_answer = {'question': self.question, 'extracted_answer': extracted_answer,  'finish': finish,
                        'real_answer': ground_truth, 'correctness': correctness, 'time': time_end - time_start}
        return final_answer
           