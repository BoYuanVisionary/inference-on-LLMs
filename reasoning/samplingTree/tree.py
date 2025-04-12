from reasoning.samplingTree.path import Path
import numpy as np



# this is the task class for the inference problem
class Tree(object):
    def __init__(self, system_prompt, question, policy_model, reward_model, sampling_method):
        self.question = question
        self.policy_model = policy_model 
        self.reward_model = reward_model
        self.system_prompt = system_prompt
        
        self.paths = [] 

        if sampling_method not in ['node', 'MH', 'stochastic_beam_search']:
            raise ValueError("Invalid sampling method")
        self.sampling_method = sampling_method
    
    def set_proposal_params_node(self, temperature = 1):
        self.temperature = temperature
    
    def generate_next_trajectory(self):

        if len(self.paths) == 0:
            current_path = None
        else:
            current_path = self.paths[-1]

        # both the new_solution and the partial solution are strings
        if self.sampling_method == 'node':
            new_solution, partial_solution = self.sampling_node(current_path) 
        elif self.sampling_method == 'stochastic_beam_search':
            new_solution, partial_solution = self.sampling_stochastic_beam_search(current_path)
        elif self.sampling_method == 'MH':
            new_solution, partial_solution = self.sampling_MH(current_path)

        # print(f'partial_solution: {partial_solution}')
        # print(f'new_solution: {new_solution}')
        solution = partial_solution + ' ' + new_solution if partial_solution is not None else new_solution
        new_path = Path(solution)
        scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)
        new_path.scores = scores
        self.paths.append(new_path)  
        return new_path           

    def sampling_node(self, path):

        if path is None:
            # print(f'path is None')
            solutions = self.policy_model.get_local_response_llama_vllm(self.system_prompt, self.question, 1)
            return solutions[-1], None
        else:
            scores = np.array(path.scores)
            # find the index of the first score that is smaller than 0.9. If the firs
            first_score_index = np.where(scores < 0.9)[0][0] if np.any(scores < 0.9) else len(scores)-1
            if first_score_index == 0:
                # print(f'first_score_index: {-1}')
                solutions = self.policy_model.get_local_response_llama_vllm(self.system_prompt, self.question, 1)
                return solutions[-1], None
            # select the node based on the score, higher score means higher chance to be selected
            scores = scores[:first_score_index+1]
            scores = np.exp(scores / self.temperature) / np.sum(np.exp(scores / self.temperature)) 

            selected_node = np.random.choice(range(len(scores)), size=1, p=scores)[0]
            # print(f'selected_node: {selected_node+1}') # selected_node is the index of the last step I would keep
            partial_solution = ' '.join(path.steps[:selected_node+1])
            solutions = self.policy_model.get_local_response_llama_vllm_completion(self.system_prompt, self.question, partial_solution, 1)
            return solutions[-1], partial_solution

    def sampling_MH(self, path):

        if path is None:
            # print(f'path is None')
            solutions = self.policy_model.get_local_response_llama_vllm(self.system_prompt, self.question, 1)
            return solutions[-1], None
        else:
            scores = np.array(path.scores)

            # sample a new solution from the policy model
            new_solution = self.policy_model.get_local_response_llama_vllm(self.system_prompt, self.question, 1)
            new_path = Path(new_solution[0])
            new_scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)

            # accept the new solution with probability min(1, new_score/mean_score)
            accpetance_prob = min(1, np.mean(np.array(new_scores))/np.mean(np.array(scores)))
            if np.random.rand() < accpetance_prob:
                print(f'accept the new solution')
                return new_solution[-1], None
            else:
                return path.solutions, None
    
    def sampling_stochastic_beam_search(self, path):
        if path is None:
            solutions = self.policy_model.get_local_response_llama_vllm(self.system_prompt, self.question, 1)
            return solutions[-1], None
        else:
            scores = np.array(path.scores)
            # resample steos based on the scores of all steps
            temperature = 0.2
            beam_width = len(scores)
            normalized_scores = np.exp(-scores/temperature) / sum(np.exp(-scores/temperature))
            # randomly sample beam_width steps based on the normalized scores
            next_steps = np.random.choice(range(len(normalized_scores)), size=beam_width, p=normalized_scores)
            # for each chosen step, sample a new solution from the policy model
            new_solutions = []
            target_score = np.min(path.scores)
            best_path = path
            for step in next_steps:
                partial_solution = ' '.join(path.steps[:step])
                solutions = self.policy_model.get_local_response_llama_vllm_completion(self.system_prompt, self.question, partial_solution, 1)
                new_solutions.append(solutions[-1])
                # reevalue each steps
                new_path = Path(partial_solution + '\n' + new_solutions[-1])
                new_scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)
                print(f'new_scores: {new_scores}')
                if np.min(new_scores) > target_score:
                    target_score = np.min(new_scores)
                    best_path = new_path
            return best_path.solutions, None

        