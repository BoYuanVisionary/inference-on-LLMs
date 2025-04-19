import numpy as np
import re
import warnings
import json
from reasoning.API.openrouter import Openrouter
from reasoning.inference.base import BaseInference
from reasoning.evaluator.math_grader import math_equal, extract_answer

#existing issue: error handling for api calls
# need to include text before step 1
class Path(object):
    def __init__(self, solutions):
        self.openrouter = Openrouter()
        self.apicalls = 0
        self.error_steps_count = 0
        
        self.steps = self.solutions_to_steps(solutions)
        if len(self.steps) == 0:
            self.steps = [solutions]
            print('problematic solutions:')
            print(f'solutions: {solutions}')
            print(f'steps: {self.steps}')
        
        self.solutions = solutions
        self.scores = None
    

    # we can also consider combine some steps if their scores are close to each other

    def solutions_to_steps(self, solutions):
        steps = []
        
        # Check if solutions is a string
        if not isinstance(solutions, str):
            raise ValueError("solutions must be a string")
        
        # Try to find numbered steps in various formats
        solution_text = solutions.strip()
        
        # First check for the pattern where steps are numbered with digits followed by periods
        # This pattern captures content between numbered items
        # numbered_pattern = re.compile(r'(?:^|\n)\s*(\d+)\.\s+(.*?)(?=(?:\n\s*\d+\.)|$)', re.DOTALL)
        # matches = numbered_pattern.findall(solution_text)
        
        # if matches:
        #     for _, step_content in matches:
        #         if step_content:
        #             steps.append(step_content.strip())
        #     if len(steps) > 0:
        #         return steps
        #     else:
        #         steps = []
            
        if '\n\n' in solution_text:
            steps = [step.strip() for step in solution_text.split('\n\n') if step.strip()]
            if len(steps) > 0:
                return steps
            else:
                steps = []
        
        if '\n' in solution_text:
            steps = [step.strip() for step in solution_text.split('\n') if step.strip()]
            if len(steps) > 0:
                return steps
            else:
                steps = []
        ###################################################
        # The following code needs further testing
        warnings.warn("using step marker pattern to decompose the solution")
        step_marker_pattern = re.compile(r'(?:^|\n)\s*#*\s*(?:S|s)tep\s+\d+:', re.DOTALL)
        matches = list(step_marker_pattern.finditer(solution_text)) # Convert to list to easily get indices

        if matches:
            steps = [] # Reset steps list for this parsing method

            # 1. Handle the preamble (content before the first marker)
            first_marker_start_index = matches[0].start()
            preamble = solution_text[0:first_marker_start_index].strip()

            # 2. Extract and combine each step segment (marker + content)
            for i, match in enumerate(matches):
                step_segment_start_index = match.start()

                step_segment_end_index = matches[i+1].start() if i + 1 < len(matches) else len(solution_text)

                segment = solution_text[step_segment_start_index:step_segment_end_index]

                if i == 0 and preamble:
                    combined_step_segment = preamble + '\n' + segment if preamble else segment
                else:
                    combined_step_segment = segment

                stripped_step = combined_step_segment.strip()
                if stripped_step: # Only add non-empty steps
                    steps.append(stripped_step)

            if steps:
                return steps
            else:
                steps = []
        ###################################################     
        warnings.warn("No explicit steps found in the solutions. Using gpt instead")
        self.apicalls += 1
        user_prompt = solutions 
    
        system_prompt = self.openrouter.set_system_prompt_for_step_decomposition()
        output = self.openrouter.completion(system_prompt, user_prompt)
        try:
            output_json = json.loads(output)
            steps = list(output_json.values())
        except:
            steps = [output]
            warnings.warn("Failed to parse the output as a json. Using the output as a single step.")
        return steps


class Tree(BaseInference):
    def __init__(self, system_prompt, question, policy_model, reward_model, sampling_method, config_name, sampling_params, sampling_temperature, beam_width, threshold, ORM_type):


        super().__init__(policy_model, policy_model.get_tokenizer(), sampling_params, config_name, reward_model, method="beam_search", ORM_type=ORM_type)
        self.question = question
        self.system_prompt = system_prompt
        self.sampling_temperature = sampling_temperature # temperature for stochastic beam search
        self.beam_width = beam_width
        self.ORM_type = ORM_type
        # In stochastic beam search, we can also store all the explored paths
        # The union of the explored paths and the paths is the set of all the paths
        self.explored_paths = [] 
        self.paths = [] 

        if sampling_method not in ['node', 'MH', 'stochastic_beam_search', 'stochastic_beam_search_version2', 'stochastic_beam_search_version3', 'stochastic_beam_search_version4']:
            raise ValueError("Invalid sampling method")
        self.sampling_method = sampling_method

        self.num_generated_tokens = 0
        self.threshold = threshold

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

    
    def generate_next_trajectory(self):

        if len(self.paths) == 0:
            current_path = None
        else:
            current_path = self.paths[-1]

        # both the new_solution and the partial solution are strings
        if self.sampling_method == 'node':
            new_path = self.sampling_node(current_path) 
        elif self.sampling_method == 'stochastic_beam_search':
            new_path = self.sampling_stochastic_beam_search(current_path)
        elif self.sampling_method == 'stochastic_beam_search_version2':
            new_path = self.sampling_stochastic_beam_search_version2(current_path)
        elif self.sampling_method == 'stochastic_beam_search_version3':
            new_path = self.sampling_stochastic_beam_search_version3(current_path)
        elif self.sampling_method == 'stochastic_beam_search_version4':
            new_path = self.sampling_stochastic_beam_search_version4(current_path)
        elif self.sampling_method == 'MH':
            new_path = self.sampling_MH(current_path)

        return new_path           

    # def sampling_node(self, path):

    #     if path is None:
    #         # print(f'path is None')
    #         solutions = self.policy_model.get_local_response_llama_vllm(self.system_prompt, self.question, 1)
    #         return solutions[-1], None
    #     else:
    #         scores = np.array(path.scores)
    #         # find the index of the first score that is smaller than 0.9. If the firs
    #         first_score_index = np.where(scores < 0.9)[0][0] if np.any(scores < 0.9) else len(scores)-1
    #         if first_score_index == 0:
    #             # print(f'first_score_index: {-1}')
    #             solutions = self.policy_model.get_local_response_llama_vllm(self.system_prompt, self.question, 1)
    #             return solutions[-1], None
    #         # select the node based on the score, higher score means higher chance to be selected
    #         scores = scores[:first_score_index+1]
    #         scores = np.exp(scores / self.temperature) / np.sum(np.exp(scores / self.temperature)) 

    #         selected_node = np.random.choice(range(len(scores)), size=1, p=scores)[0]
    #         # print(f'selected_node: {selected_node+1}') # selected_node is the index of the last step I would keep
    #         partial_solution = ' '.join(path.steps[:selected_node+1])
    #         solutions = self.policy_model.get_local_response_llama_vllm_completion(self.system_prompt, self.question, partial_solution, 1)
    #         return solutions[-1], partial_solution

    # def sampling_MH(self, path):

    #     if path is None:
    #         # print(f'path is None')
    #         solutions = self.policy_model.get_local_response_llama_vllm(self.system_prompt, self.question, 1)
    #         new_path = Path(solutions[-1])
    #         new_path.scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)
    #         return new_path
    #     else:
    #         scores = np.array(path.scores)

    #         # sample a new solution from the policy model
    #         new_solution = self.policy_model.get_local_response_llama_vllm(self.system_prompt, self.question, 1)
    #         new_path = Path(new_solution[0])
    #         new_scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)

    #         # accept the new solution with probability min(1, new_score/mean_score)
    #         accpetance_prob = min(1, np.mean(np.array(new_scores))/np.mean(np.array(scores)))
    #         if np.random.rand() < accpetance_prob:
    #             print(f'accept the new solution')
    #             return new_path
    #         else:
    #             return path
    
    # def sampling_stochastic_beam_search(self, path): 
    #     if path is None:
    #         solutions, num_generated_tokens, num_input_tokens = self.generate_text(self.system_prompt, [self.question])
    #         self.num_generated_tokens += sum(num_generated_tokens)
    #         new_path = Path(solutions[-1])
    #         # In principle, we should add the new path to the explored paths
    #         # But for a fair comparison with the other methods, we do not add it to the explored paths
    #         # self.explored_paths.append(new_path) 
    #         new_path.scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)
    #         return new_path
    #     else:
    #         scores = np.array(path.scores)
    #         # resample based on the scores of all steps
    #         temperature = self.sampling_temperature
    #         beam_width = len(scores) if self.beam_width is None else self.beam_width
    #         normalized_scores = np.exp(-scores/temperature) / sum(np.exp(-scores/temperature))
    #         # randomly sample beam_width steps based on the normalized scores
    #         next_steps = np.random.choice(range(len(normalized_scores)), size=beam_width, p=normalized_scores)
    #         # for each chosen step, sample a new solution from the policy model
    #         new_solutions = []
    #         target_score = np.min(path.scores)
    #         best_path = path

    #         partial_solutions = []
    #         for step_idx in next_steps:
    #             formatted_steps = []
    #             for i, step_content in enumerate(path.steps[:step_idx]):
    #                 formatted_steps.append(f"## Step {i+1}: {step_content}")
    #             partial_solutions.append('\n\n'.join(formatted_steps))
          

    #         questions = [self.question] * len(next_steps)
    #         new_solutions, num_generated_tokens, num_input_tokens = self.generate_text_completion(self.system_prompt, questions, partial_solutions)   
    #         # print(f'new_solutions: {new_solutions}')
    #         self.num_generated_tokens += sum(num_generated_tokens)

    #         for i in range(len(next_steps)): # works only when the last parameter of get_local_response_llama_vllm_completion_batch is 1
    #             new_path = Path(partial_solutions[i] + '\n' + new_solutions[i])
    #             self.explored_paths.append(new_path)
    #             new_scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)
    #             new_path.scores = new_scores
    #             # for score, step in zip(new_scores, new_path.steps):
    #             #     print(f'score: {score}, step: {step}')
    #             if np.min(new_scores) > target_score:
    #                 target_score = np.min(new_scores)
    #                 best_path = new_path
    #         print(f'best_path: {best_path.steps}')
    #         return best_path
        
    # def sampling_stochastic_beam_search_version2(self, path): # added a filtering mechanism to the original stochastic beam search
        
    #     if path is None:
    #         solutions, num_generated_tokens, num_input_tokens = self.generate_text(self.system_prompt, [self.question])
    #         self.num_generated_tokens += sum(num_generated_tokens)
    #         new_path = Path(solutions[-1])
    #         # In principle, we should add the new path to the explored paths
    #         # But for a fair comparison with the other methods, we do not add it to the explored paths
    #         # self.explored_paths.append(new_path) 
    #         new_path.scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)
    #         return new_path
    #     else:
    #         scores = np.array(path.scores)
    #         # resample based on the scores of all steps
    #         temperature = self.sampling_temperature
    #         beam_width = len(scores) if self.beam_width is None else self.beam_width
    #         # find the first index where the score is smaller than the threshold (default is 0.9)
    #         first_score_index = np.where(scores < self.threshold)[0][0] if np.any(scores < self.threshold) else -1
    #         if first_score_index == -1:
    #             for _ in range(beam_width):
    #                 self.explored_paths.append(path)
    #             return path
    #         else:
    #             scores = scores[:first_score_index+1]
    #             normalized_scores = np.exp(-scores/temperature) / sum(np.exp(-scores/temperature))
    #         # randomly sample beam_width steps based on the normalized scores
    #         next_steps = np.random.choice(range(len(normalized_scores)), size=beam_width, p=normalized_scores)
    #         # for each chosen step, sample a new solution from the policy model
    #         new_solutions = []
    #         target_score = np.min(path.scores)
    #         best_path = path

    #         # partial_solutions = [' '.join(path.steps[:step]) for step in next_steps]
    #         # print(f'partial_solutions: {partial_solutions}')

    #         partial_solutions = []
    #         for step_idx in next_steps:
    #             formatted_steps = []
    #             for i, step_content in enumerate(path.steps[:step_idx]):
    #                 formatted_steps.append(f"## Step {i+1}: {step_content}")
    #             partial_solutions.append('\n\n'.join(formatted_steps))

    #         questions = [self.question] * len(next_steps)
    #         new_solutions, num_generated_tokens, num_input_tokens = self.generate_text_completion(self.system_prompt, questions, partial_solutions)   
    #         # print(f'new_solutions: {new_solutions}')
    #         self.num_generated_tokens += sum(num_generated_tokens)

    #         for i in range(len(next_steps)): # works only when the last parameter of get_local_response_llama_vllm_completion_batch is 1
    #             new_path = Path(partial_solutions[i] + '\n' + new_solutions[i])
    #             self.explored_paths.append(new_path)
    #             new_scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)
    #             new_path.scores = new_scores
    #             # for score, step in zip(new_scores, new_path.steps):
    #             #     print(f'score: {score}, step: {step}')
    #             if np.min(new_scores) > target_score:
    #                 target_score = np.min(new_scores)
    #                 best_path = new_path
    #         print(f'best_path: {best_path.steps}')
    #         return best_path
        
    # def sampling_stochastic_beam_search_version3(self, path):
    #     if path is None:
    #         solutions, num_generated_tokens, num_input_tokens = self.generate_text(self.system_prompt, [self.question])
    #         self.num_generated_tokens += sum(num_generated_tokens)
    #         new_path = Path(solutions[-1])
    #         # In principle, we should add the new path to the explored paths
    #         # But for a fair comparison with the other methods, we do not add it to the explored paths
    #         # self.explored_paths.append(new_path) 
    #         new_path.scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)
    #         return new_path
    #     else:
    #         scores = np.array(path.scores)
    #         # resample based on the scores of all steps
    #         temperature = self.sampling_temperature
    #         beam_width = len(scores) if self.beam_width is None else self.beam_width # consider how to make it adaptive

    #         augmented_scores = np.concatenate(([1.0], scores))
    #         score_diffs = np.diff(augmented_scores)
    #         # Only consider indices where the score is decreasing (negative difference)
    #         decreasing_indices = np.where(score_diffs < 0)[0]
    #         if len(decreasing_indices) == 0: # the only possible case is that all scores are 1.
    #             assert np.all(scores == 1.0)
    #             for _ in range(beam_width):
    #                 self.explored_paths.append(path)
    #             return path
            
    #         else:                
    #             decrease_magnitudes = -score_diffs[decreasing_indices]
    #             normalized_magnitudes = decrease_magnitudes/temperature / sum(decrease_magnitudes/temperature) # exp is not a good choice
                
    #             sampled_indices = np.random.choice(len(decreasing_indices), size=beam_width, p=normalized_magnitudes)
                
    #             # Get the actual step indices (add 1 because decreasing_indices refers to the augmented array)
    #             next_steps = decreasing_indices[sampled_indices]
    #             new_solutions = []
    #             target_score = np.min(path.scores)
    #             best_path = path
    #             # If next_step is 0, the first step in the orignal path already has a drop in score
    #             # So just remove all existing steps and start from the beginning
    #             # partial_solutions = [' '.join(path.steps[:step]) for step in next_steps]
    #             # print(f'partial_solutions: {partial_solutions}')
    #             partial_solutions = []
    #             for step_idx in next_steps:
    #                 formatted_steps = []
    #                 for i, step_content in enumerate(path.steps[:step_idx]):
    #                     formatted_steps.append(f"## Step {i+1}: {step_content}")
    #                 partial_solutions.append('\n\n'.join(formatted_steps))

    #             questions = [self.question] * len(next_steps)
    #             new_solutions, num_generated_tokens, num_input_tokens = self.generate_text_completion(self.system_prompt, questions, partial_solutions)   
    #             # print(f'new_solutions: {new_solutions}')
    #             self.num_generated_tokens += sum(num_generated_tokens)

    #             for i in range(len(next_steps)): # works only when the last parameter of get_local_response_llama_vllm_completion_batch is 1
    #                 new_path = Path(partial_solutions[i] + '\n\n' + new_solutions[i]) # needed if new_solution is also part of the partial solution
    #                 self.explored_paths.append(new_path)
    #                 new_scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)
    #                 new_path.scores = new_scores
    #                 # for score, step in zip(new_scores, new_path.steps):
    #                 #     print(f'score: {score}, step: {step}')
    #                 if np.min(new_scores) > target_score:
    #                     target_score = np.min(new_scores)
    #                     best_path = new_path
    #             print(f'best_path: {best_path.steps}')
    #             print(f'best_path.scores: {best_path.scores}')
    #             print(f'best_path.scores.min: {np.min(best_path.scores)}')
    #             return best_path


    def sampling_stochastic_beam_search_version4(self, path):
        # find the first index where the score is smaller than the threshold (default is 0.9). And just do sampling on it. 
        # using weighted BON at each step
        if path is None:
            solutions, num_generated_tokens, num_input_tokens = self.generate_text(self.system_prompt, [self.question])
            self.num_generated_tokens += sum(num_generated_tokens)
            new_path = Path(solutions[-1])
            # In principle, we should add the new path to the explored paths
            # But for a fair comparison with the other methods, we do not add it to the explored paths
            # self.explored_paths.append(new_path) 
            new_path.scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)
            return new_path
        else:
            scores = np.array(path.scores)
            # resample based on the scores of all steps
            beam_width = len(scores) if self.beam_width is None else self.beam_width
            first_score_index = np.where(scores < self.threshold)[0][0] if np.any(scores < self.threshold) else -1
            print(f'first_score_index: {first_score_index}')
            print(f'scores: {scores}')
            print(f'path.steps: {path.steps}')
            if first_score_index == -1: # if all scores are larger than the threshold, just return the original path
                for _ in range(beam_width):
                    self.explored_paths.append(path)
                return path
    
            next_steps = [first_score_index] * beam_width
            # for each chosen step, sample a new solution from the policy model
            new_solutions = []
            # target_score = np.min(path.scores) # or maybe the score of current node
            # best_path = path

            partial_solutions = []
            for step_idx in next_steps:
                formatted_steps = []
                for i, step_content in enumerate(path.steps[:step_idx]):
                    # formatted_steps.append(f"## Step {i+1}: {step_content}")
                    formatted_steps.append(step_content)
                partial_solutions.append('\n\n'.join(formatted_steps))
          
            # consider adding new step based on the previous feedback
            questions = [self.question] * len(next_steps)
            new_solutions, num_generated_tokens, num_input_tokens = self.generate_text_completion(self.system_prompt, questions, partial_solutions)   
            # print(f'new_solutions: {new_solutions}')
            self.num_generated_tokens += sum(num_generated_tokens)
            all_scores = []
            extracted_answers = []
            new_paths = []
            assert len(next_steps) > 1
            for i in range(len(next_steps)): # works only when the last parameter of get_local_response_llama_vllm_completion_batch is 1
                
                new_path = Path(partial_solutions[i] + '\n\n' + new_solutions[i])

                # if len(partial_solutions[i]) > 5: # to ensure the partial solution is not too short like '' or '.' or '\n'
                #     temp_path1 = Path(partial_solutions[i])
                #     temp_path2 = Path(new_solutions[i])
                #     new_path.steps = temp_path1.steps + temp_path2.steps 
                new_paths.append(new_path)

                self.explored_paths.append(new_path)
                new_scores = self.reward_model.get_value_with_steps(self.question, new_path.steps)
                new_path.scores = new_scores
                all_scores.append(self.get_reward(new_scores))
                extracted_answers.append(extract_answer(new_path.solutions))                
                # Check if scores are significantly different (threshold of 0.01)
                score_diff = np.abs(np.array(new_path.scores[:first_score_index]) - np.array(path.scores[:first_score_index]))
                if np.any(score_diff > 0.1):
                    warnings.warn(f'The first {first_score_index} scores differ significantly between paths')
                    print(f'Original scores: {path.scores[:first_score_index]}')
                    print(f'New scores: {new_path.scores[:first_score_index]}')

                
            # self consistency mechanism
            weighted_votes = [0.0] * len(next_steps)
            
            for i in range(len(extracted_answers)):
                for j in range(len(extracted_answers)):
                    if math_equal(extracted_answers[i], extracted_answers[j]):
                        weighted_votes[i] += all_scores[j]
            
            selected_path = new_paths[max(range(len(weighted_votes)), key=lambda i: weighted_votes[i])]
            return selected_path



if __name__ == "__main__":


    solutions = """
    To solve the equation \\( x = \\sqrt{11 - 2x} + 4 \\), we will follow these steps:\n\n1. **Isolate the square root term:**\n   \\[\n   x - 4 = \\sqrt{11 - 2x}\n   \\]\n\n2. **Square both sides to eliminate the square root:**\n   \\[\n   (x - 4)^2 = (\\sqrt{11 - 2x})^2\n   \\]\n   Simplifying both sides, we get:\n   \\[\n   (x - 4)^2 = 11 - 2x\n   \\]\n\n3. **Expand the left-hand side:**\n   \\[\n   x^2 - 8x + 16 = 11 - 2x\n   \\]\n\n4. **Move all terms to one side to form a quadratic equation:**\n   \\[\n   x^2 - 8x + 16 - 11 + 2x = 0\n   \\]\n   Simplifying, we get:\n   \\[\n   x^2 - 6x + 5 = 0\n   \\]\n\n5. **Factor the quadratic equation:**\n   \\[\n   (x - 1)(x - 5) = 0\n   \\]\n\n6. **Solve for \\( x \\):**\n   \\[\n   x - 1 = 0 \\quad \\text{or} \\quad x - 5 = 0\n   \\]\n   \\[\n   x = 1 \\quad \\text{or} \\quad x = 5\n   \\]\n\n7. **Verify the solutions by substituting them back into the original equation:**\n\n   - For \\( x = 1 \\):\n     \\[\n     1 = \\sqrt{11 - 2(1)} + 4\n     \\]\n     \\[\n     1 = \\sqrt{9} + 4\n     \\]\n     \\[\n     1 = 3 + 4\n     \\]\n     \\[\n     1 \\neq 7\n     \\]\n     Therefore, \\( x = 1 \\) is not a solution.\n\n   - For \\( x = 5 \\):\n     \\[\n     5 = \\sqrt{11 - 2(5)} + 4\n     \\]\n     \\[\n     5 = \\sqrt{1} + 4\n     \\]\n     \\[\n     5 = 1 + 4\n     \\]\n     \\[\n     5 = 5\n     \\]\n     Therefore, \\( x = 5 \\) is a solution.\n\nThus, the only value of \\( x \\) that satisfies the equation is \\(\\boxed{5}\\).
    """
    solutions = """To determine the number of positive whole-number divisors of 196, we start by finding its prime factorization.\n\n1. **Prime Factorization of 196:**\n   - First, we check if 196 is divisible by the smallest prime number, 2.\n   - \\(196 \\div 2 = 98\\), so 196 is divisible by 2.\n   - Next, we factor 98: \\(98 \\div 2 = 49\\), so 98 is also divisible by 2.\n   - Now, we factor 49: \\(49 = 7 \\times 7\\), so 49 is divisible by 7.\n   - Therefore, the prime factorization of 196 is \\(2^2 \\times 7^2\\).\n\n2. **Using the Prime Factorization to Find the Number of Divisors:**\n   - If a number \\(n\\) has a prime factorization of the form \\(p_1^{e_1} \\times p_2^{e_2} \\times \\cdots \\times p_k^{e_k}\\), then the number of positive divisors of \\(n\\) is given by \\((e_1 + 1)(e_2 + 1) \\cdots (e_k + 1)\\).\n   - For 196, the prime factorization is \\(2^2 \\times 7^2\\).\n   - Here, \\(e_1 = 2\\) and \\(e_2 = 2\\).\n   - Applying the formula, the number of divisors is \\((2 + 1)(2 + 1) = 3 \\times 3 = 9\\).\n\nTherefore, the number of positive whole-number divisors of 196 is \\(\\boxed{9}\\).    """
    solutions = """To solve the problem \\(1 - 2 + 3 - 4 + 5 - \\dots + 99 - 100\\), we will follow a step-by-step approach.\n\nFirst, let's observe the pattern in the series:\n\\[1 - 2 + 3 - 4 + 5 - \\dots + 99 - 100\\]\n\nWe can group the terms in pairs:\n\\[(1 - 2) + (3 - 4) + (5 - 6) + \\dots + (99 - 100)\\]\n\nEach pair can be simplified:\n\\[1 - 2 = -1\\]\n\\[3 - 4 = -1\\]\n\\[5 - 6 = -1\\]\n\\[\\vdots\\]\n\\[99 - 100 = -1\\]\n\nNow, we need to determine how many such pairs there are. The sequence from 1 to 100 has 100 terms. Since we are pairing them, the number of pairs is:\n\\[\\frac{100}{2} = 50\\]\n\nEach pair sums to \\(-1\\), so the total sum of all pairs is:\n\\[50 \\times (-1) = -50\\]\n\nThus, the final answer is:\n\\[\\boxed{-50}\\]"""
    solutions = "## Step 1: Convert the numbers to base 10\nTo find the product of $6_8$ and $7_8$, we first need to convert these numbers to base 10. $6_8$ in base 10 is $6 \\times 8^0 = 6$. $7_8$ in base 10 is $7 \\times 8^0 = 7$.\n\n## Step 2: Multiply the numbers in base 10\nNow we multiply the two numbers in base 10. $6 \\times 7 = 42$.\n\n## Step 3: Convert the product back to base 8\nTo convert 42 to base 8, we divide it by 8 and keep track of the remainders. $42 \\div 8 = 5$ remainder 2. So, the number in base 8 is $52_8$.\n\nThe final answer is: $\\boxed{52_8}$"
    
    path = Path(solutions)
    steps = path.steps
    for step in steps:
        print("-"*100)
        print(step)
    
    # ######################################################################################################
    # with open("/ssdscratch/byuan48/efficient_reasoning/results/Qwen2.5-7B-Instruct-greedy.jsonl", "r") as f:
    #     items = [json.loads(line) for line in f]
    # solutions = [item['generated_solutions'][0] for item in items]

    # path = Path()

    # for solution in solutions:     
    #     print(solution)
    #     steps = path.solutions_to_steps(solution)
    #     for step in steps:
    #         print("-"*100)
    #         print(step)
    # print(f"Total API calls: {path.apicalls}")        