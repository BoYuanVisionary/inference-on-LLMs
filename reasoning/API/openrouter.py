from openai import OpenAI
import json


class Openrouter:
    def __init__(self, model="openai/gpt-4o-2024-11-20"):
        self.client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
        )
        self.model = model

    def completion(self, system_prompt, user_prompt):

        completion = self.client.chat.completions.create(
        extra_body={},
        model=self.model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": user_prompt
                },
            ]
            }
        ],
        response_format={ "type": "json_object" }
        )
        # print(completion)
        return completion.choices[0].message.content
    
    def set_system_prompt_for_step_decomposition(self):
        system_prompt = """Your task is to decompose reasoning steps (sentences) into individual steps. Treat this task as text processing rather than analytical reasoning. Follow these specific rules:

        Do not analyze or interpret the reasoning steps; simply decompose them as they are given.

        If the input already contains explicit decomposition, such as "Step 1, Step 2, ..." or numbered steps (e.g., "1., 2., 3."), retain and follow the original structure.

        If the input does not have explicit steps, decompose it using the logical structure of the reasoning.

        Your primary goal is decomposition – maintain as much of the original text as possible without modification.

        If there is content before the first step, include it as part of the first step without altering its order.

        You don't need to decompose sentence by sentence. Some steps are strongly related and should be kept together as a single step when they form a coherent reasoning unit.

        The output should be in JSON format, where each line represents an individual reasoning step."""
        self.system_prompt = system_prompt
        return system_prompt

    def format_steps_to_arrays(self, steps_list):
        """
        Convert a list of steps into a structured format with arrays of lines.
        
        Args:
            steps_list (list): A list of strings where each string is one step
            
        Returns:
            list: A list of dictionaries, each with 'content' (original step) and 'lines' (array of lines)
        """
        steps_with_arrays = []
        
        for i, step_content in enumerate(steps_list, 1):
            # Split the step content into lines and filter out empty ones
            lines = [line.strip() for line in step_content.split('\n') if line.strip()]
            
            # Store both the original content and the array of lines
            steps_with_arrays.append({
                "step_number": i,
                "content": step_content,
                "lines": lines
            })
            
        return steps_with_arrays
        
    def list_to_string(self, string_list, separator="\n\n"):
        """
        Converts a list of strings to a single concatenated string.
        
        Args:
            string_list (list): List of strings to concatenate
            separator (str, optional): Separator to use between strings. Defaults to "\n\n".
            
        Returns:
            str: Concatenated string
        """
        return separator.join(string_list)


if __name__ == "__main__":
    openrouter = Openrouter()
    system_prompt = """Your task is to decompose reasoning steps (sentences) into individual steps. Treat this task as text processing rather than analytical reasoning. Follow these specific rules:

    Do not analyze or interpret the reasoning steps; simply decompose them as they are given.

    If the input already contains explicit decomposition, such as "Step 1, Step 2, ..." or numbered steps (e.g., "1., 2., 3."), retain and follow the original structure.

    If the input does not have explicit steps, decompose it using the logical structure of the reasoning.

    Your primary goal is decomposition – maintain as much of the original text as possible without modification.

    If there is content before the first step, include it as part of the first step without altering its order.

    You don't need to decompose sentence by sentence. Some steps are strongly related and should be kept together as a single step when they form a coherent reasoning unit.

    The output should be in JSON format, where each line represents an individual reasoning step."""

    user_prompt = """ To solve the problem \\(1 - 2 + 3 - 4 + 5 - \\dots + 99 - 100\\), we will follow a step-by-step approach.\n\nFirst, let's observe the pattern in the series:\n\\[1 - 2 + 3 - 4 + 5 - \\dots + 99 - 100\\]\n\nWe can group the terms in pairs:\n\\[(1 - 2) + (3 - 4) + (5 - 6) + \\dots + (99 - 100)\\]\n\nEach pair can be simplified:\n\\[1 - 2 = -1\\]\n\\[3 - 4 = -1\\]\n\\[5 - 6 = -1\\]\n\\[\\vdots\\]\n\\[99 - 100 = -1\\]\n\nNow, we need to determine how many such pairs there are. The sequence from 1 to 100 has 100 terms. Since we are pairing them, the number of pairs is:\n\\[\\frac{100}{2} = 50\\]\n\nEach pair sums to \\(-1\\), so the total sum of all pairs is:\n\\[50 \\times (-1) = -50\\]\n\nThus, the final answer is:\n\\[\\boxed{-50}\\]
    """
    output = openrouter.completion(system_prompt, user_prompt)
    # load the output as a json object
    output_json = json.loads(output)
    # print(output_json)

    # this is the expected output
    # {
    # "step_1": "To solve the problem \\(1 - 2 + 3 - 4 + 5 - \\dots + 99 - 100\\), we will follow a step-by-step approach.",
    # "step_2": "First, let's observe the pattern in the series: \\[1 - 2 + 3 - 4 + 5 - \\dots + 99 - 100\\]",
    # "step_3": "We can group the terms in pairs: \\[(1 - 2) + (3 - 4) + (5 - 6) + \\dots + (99 - 100)\\]",
    # "step_4": "Each pair can be simplified: \\[1 - 2 = -1\\] \\[3 - 4 = -1\\] \\[5 - 6 = -1\\] \\[\\vdots\\] \\[99 - 100 = -1\\]",
    # "step_5": "Now, we need to determine how many such pairs there are. The sequence from 1 to 100 has 100 terms.",
    # "step_6": "Since we are pairing them, the number of pairs is: \\[\\frac{100}{2} = 50\\]",
    # "step_7": "Each pair sums to \\(-1\\), so the total sum of all pairs is: \\[50 \\times (-1) = -50\\]",
    # "step_8": "Thus, the final answer is: \\[\\boxed{-50}\\]"
    # }
    
