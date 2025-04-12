import re
import warnings
import json
from reasoning.API.openrouter import Openrouter

#existing issue: error handling for api calls
class Path(object):
    def __init__(self, solutions):
        self.openrouter = Openrouter()
        self.apicalls = 0

        self.steps = self.solutions_to_steps(solutions)
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
        numbered_pattern = re.compile(r'(?:^|\n)\s*(\d+)\.\s+(.*?)(?=(?:\n\s*\d+\.)|$)', re.DOTALL)
        matches = numbered_pattern.findall(solution_text)
        
        if matches:
            for _, step_content in matches:
                if step_content:
                    steps.append(step_content.strip())
            return steps
            
    # If no numbered steps found, try the "Step X:" or "step X:" format
        step_pattern = re.compile(r'(?:^|\n)\s*#{1,6}\s+(?:S|s)tep\s+\d+:\s*(.*?)(?=(?:\n\s*#{1,6}\s+(?:S|s)tep\s+\d+:)|$)', re.DOTALL)
        matches = step_pattern.finditer(solution_text)
        if matches:
            for match in matches:
                step_content = match.group(1).strip()
                if step_content:
                    steps.append(step_content)
            return steps
        else:# if no steps found, use gpt to decompose the solutions
            warnings.warn("No explicit steps found in the solutions. Using gpt instead")
            self.apicalls += 1
            user_prompt = solutions 
            system_prompt = self.openrouter.set_system_prompt_for_step_decomposition()
            output = self.openrouter.completion(system_prompt, user_prompt)
            # print(output)
            try:
                output_json = json.loads(output)
                steps = list(output_json.values())
            except:
                steps = [output]
                warnings.warn("Failed to parse the output as a json. Using the output as a single step.")
            return steps
        


if __name__ == "__main__":


    # solutions = """
    # To solve the equation \\( x = \\sqrt{11 - 2x} + 4 \\), we will follow these steps:\n\n1. **Isolate the square root term:**\n   \\[\n   x - 4 = \\sqrt{11 - 2x}\n   \\]\n\n2. **Square both sides to eliminate the square root:**\n   \\[\n   (x - 4)^2 = (\\sqrt{11 - 2x})^2\n   \\]\n   Simplifying both sides, we get:\n   \\[\n   (x - 4)^2 = 11 - 2x\n   \\]\n\n3. **Expand the left-hand side:**\n   \\[\n   x^2 - 8x + 16 = 11 - 2x\n   \\]\n\n4. **Move all terms to one side to form a quadratic equation:**\n   \\[\n   x^2 - 8x + 16 - 11 + 2x = 0\n   \\]\n   Simplifying, we get:\n   \\[\n   x^2 - 6x + 5 = 0\n   \\]\n\n5. **Factor the quadratic equation:**\n   \\[\n   (x - 1)(x - 5) = 0\n   \\]\n\n6. **Solve for \\( x \\):**\n   \\[\n   x - 1 = 0 \\quad \\text{or} \\quad x - 5 = 0\n   \\]\n   \\[\n   x = 1 \\quad \\text{or} \\quad x = 5\n   \\]\n\n7. **Verify the solutions by substituting them back into the original equation:**\n\n   - For \\( x = 1 \\):\n     \\[\n     1 = \\sqrt{11 - 2(1)} + 4\n     \\]\n     \\[\n     1 = \\sqrt{9} + 4\n     \\]\n     \\[\n     1 = 3 + 4\n     \\]\n     \\[\n     1 \\neq 7\n     \\]\n     Therefore, \\( x = 1 \\) is not a solution.\n\n   - For \\( x = 5 \\):\n     \\[\n     5 = \\sqrt{11 - 2(5)} + 4\n     \\]\n     \\[\n     5 = \\sqrt{1} + 4\n     \\]\n     \\[\n     5 = 1 + 4\n     \\]\n     \\[\n     5 = 5\n     \\]\n     Therefore, \\( x = 5 \\) is a solution.\n\nThus, the only value of \\( x \\) that satisfies the equation is \\(\\boxed{5}\\).
    # """
    # solutions = """To determine the number of positive whole-number divisors of 196, we start by finding its prime factorization.\n\n1. **Prime Factorization of 196:**\n   - First, we check if 196 is divisible by the smallest prime number, 2.\n   - \\(196 \\div 2 = 98\\), so 196 is divisible by 2.\n   - Next, we factor 98: \\(98 \\div 2 = 49\\), so 98 is also divisible by 2.\n   - Now, we factor 49: \\(49 = 7 \\times 7\\), so 49 is divisible by 7.\n   - Therefore, the prime factorization of 196 is \\(2^2 \\times 7^2\\).\n\n2. **Using the Prime Factorization to Find the Number of Divisors:**\n   - If a number \\(n\\) has a prime factorization of the form \\(p_1^{e_1} \\times p_2^{e_2} \\times \\cdots \\times p_k^{e_k}\\), then the number of positive divisors of \\(n\\) is given by \\((e_1 + 1)(e_2 + 1) \\cdots (e_k + 1)\\).\n   - For 196, the prime factorization is \\(2^2 \\times 7^2\\).\n   - Here, \\(e_1 = 2\\) and \\(e_2 = 2\\).\n   - Applying the formula, the number of divisors is \\((2 + 1)(2 + 1) = 3 \\times 3 = 9\\).\n\nTherefore, the number of positive whole-number divisors of 196 is \\(\\boxed{9}\\).    """
    # solutions = """To solve the problem \\(1 - 2 + 3 - 4 + 5 - \\dots + 99 - 100\\), we will follow a step-by-step approach.\n\nFirst, let's observe the pattern in the series:\n\\[1 - 2 + 3 - 4 + 5 - \\dots + 99 - 100\\]\n\nWe can group the terms in pairs:\n\\[(1 - 2) + (3 - 4) + (5 - 6) + \\dots + (99 - 100)\\]\n\nEach pair can be simplified:\n\\[1 - 2 = -1\\]\n\\[3 - 4 = -1\\]\n\\[5 - 6 = -1\\]\n\\[\\vdots\\]\n\\[99 - 100 = -1\\]\n\nNow, we need to determine how many such pairs there are. The sequence from 1 to 100 has 100 terms. Since we are pairing them, the number of pairs is:\n\\[\\frac{100}{2} = 50\\]\n\nEach pair sums to \\(-1\\), so the total sum of all pairs is:\n\\[50 \\times (-1) = -50\\]\n\nThus, the final answer is:\n\\[\\boxed{-50}\\]"""
    # solutions = "## Step 1: Convert the numbers to base 10\nTo find the product of $6_8$ and $7_8$, we first need to convert these numbers to base 10. $6_8$ in base 10 is $6 \\times 8^0 = 6$. $7_8$ in base 10 is $7 \\times 8^0 = 7$.\n\n## Step 2: Multiply the numbers in base 10\nNow we multiply the two numbers in base 10. $6 \\times 7 = 42$.\n\n## Step 3: Convert the product back to base 8\nTo convert 42 to base 8, we divide it by 8 and keep track of the remainders. $42 \\div 8 = 5$ remainder 2. So, the number in base 8 is $52_8$.\n\nThe final answer is: $\\boxed{52_8}$"
    
    # path = Path(solutions)
    # steps = path.steps
    # for step in steps:
    #     print("-"*100)
    #     print(step)
    
    ######################################################################################################
    with open("/ssdscratch/byuan48/efficient_reasoning/results/Qwen2.5-7B-Instruct-greedy.jsonl", "r") as f:
        items = [json.loads(line) for line in f]
    solutions = [item['generated_solutions'][0] for item in items]

    path = Path()

    for solution in solutions:     
        print(solution)
        steps = path.solutions_to_steps(solution)
        for step in steps:
            print("-"*100)
            print(step)
    print(f"Total API calls: {path.apicalls}")
