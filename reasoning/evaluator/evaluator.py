from typing import Union
from .math_grader import math_equal, extract_answer

class Evaluator:
    def __init__(self, include_percentage: bool = True, tolerance: float = 1e-4, timeout: float = 10.0):
        self.include_percentage = include_percentage
        self.tolerance = tolerance
        self.timeout = timeout
        self.samples_without_boxed_answers = []

    def evaluate(self, prediction: Union[bool, float, str], reference: Union[float, str]) -> bool:

        attempt = extract_answer(prediction)
        answer = extract_answer(reference)
        if attempt is None or answer is None:
            self.samples_without_boxed_answers.append((prediction, reference))
            return "unknown"
        else:
            return 'yes' if math_equal(attempt, answer, self.include_percentage, self.tolerance, self.timeout) else 'no'
    
    def error_show(self,is_output=False):
        
        if len(self.samples_without_boxed_answers) == 0:
            print("No samples without boxed answers.")
            return
        else:
            print(f"Number of samples without boxed answers: {len(self.samples_without_boxed_answers)}")
            print("")
            if is_output:
                for i in range(len(self.samples_without_boxed_answers)):
                    print(f"Prediction: {self.samples_without_boxed_answers[i][0]}")
                    print(f"Reference: {self.samples_without_boxed_answers[i][1]}")
                    print("")

