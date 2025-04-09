from reasoning.SamplingTree.path import Path
import numpy as np



# this is the task class for the inference problem
class Tree(object):
    def __init__(self, system_prompt, question, policy_model, reward_model, sampling_method):
        self.question = question
        self.policy_model = policy_model 
        self.reward_model = reward_model
        self.system_prompt = system_prompt
        
        self.paths = [] 

        if sampling_method not in ['node', 'beam_search']:
            raise ValueError("Invalid sampling method")
        self.sampling_method = sampling_method
    
    def set_proposal_params_node(self, temperature):
        self.temperature = temperature
    
    def generate_next_trajectory(self):

        if len(self.paths) == 0:
            current_path = None
            query_prompt = self.question
        else:
            current_path = self.paths[-1]
            query_prompt = current_path.steps + self.question

        solution = self.policy_model.generate_trajectory(query_prompt) # generate the first trajectory
        new_path = Path(solution)
        scores = self.reward_model.get_scores(self.question, new_path.steps)
        new_path.scores = scores
        self.paths.append(new_path)  
        return new_path           

    def sampling_node(self, path):
        if path is None:
            return self.policy_model.get_local_response_llama_vllm(self.system_prompt, self.question, 1)
        else:
            # select the node based on the score, higher score means higher chance to be selected
            scores = path.scores
            # normalize the scores with softmax
            scores = np.exp(scores / self.temperature) / np.sum(np.exp(scores / self.temperature)) 
            # select the node based on the score
            selected_node = np.random.choice(range(len(scores)), size=1, p=scores)
            # regenerate the whole path

        
        
        
        