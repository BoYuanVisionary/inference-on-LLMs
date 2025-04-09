cot_prompt_en = '''
Given a science problem, your task is to answer the question step-by-step in a clear and specific manner.
The format of the solution is limited to: "Solution: ...\nSummary: The final answer is $...$"
Please complete the answer step-by-step, and finally outline the final answer.
Problem: '''

MATH_cot_prompt = '''
You are supposed to provide a solution to a given problem.\n\n
Problem:\n{query}\nSolution: Let's think step by step.\n
'''

zero_single_proposal_prompt_en = '''
Your task is to give the correct next step, given a science problem and an existing partial solution (not a complete answer).
Assuming the input is n-steps, then the format of the input is:
"Problem: ...
Existing Steps:
Step 1: ...
Step 2: ...
...
Step n: ..."
where ... denotes omitted input information.
If no existing steps are provided, you need to briefly analyze the problem from scratch and then output the first step. Otherwise, you need to output the next step (step n+1) that you think is correct, following the ideas and analysis of the existing steps.
The output format is limited to:
"Next step: ..."
where ... indicates omitted output information, which is the part you should fill in. Your output should be a complete reasoning step that includes calculations, reasoning, choosing answers, etc.
Here is the input, please follow the restricted output format.

Problem: '''

zero_single_proposal_prompt_mistral = '''
Given a science problem and an existing incomplete solution, your task is to complete the solution in a smooth and proper way.

- If no existing steps are provided, you must briefly analyze the problem and output only the first step.  
- If existing steps are sufficent to solve the problem, you must output the final answer and format the answer inside `\\boxed{}` (e.g., `\\boxed{42}`). 
- If existing steps are not sufficent to solve the problem, you must output exactly **one** correct next step that naturally follows from the previous ones.  
- You **must** follow the given format.

**Strict Output Format:**  
- Your response must always start with: `Next step: ...`  
- The response must be limited to one reasoning step (e.g., a calculation, reasoning, or answer choice).  


If there are multiple reasonable next steps, choose the most natural one based on the provided existing steps.  

'''

zero_single_proposal_prompt_gpt_en = '''
Given a science problem, you need to answer the problem based on your existing knowledge. The input may include some existing steps to solve the question and you should continue to complete the solution based on these existing steps.

If the input does not provide any existing steps, you need to analyze the problem and then give the first step in solving or calculating the problem. If partial solution steps are provided, you need to output the next step along the lines of the existing steps.
The output format is limited to: "Next step: ..."
where ... indicates omitted output information, which is the next step in the answer that you should give. Your output must be a complete reasoning step, which should include detailed calculations, reasoning, choosing answers, etc.
Below is the input, please follow the specified format for your output.

Problem: '''


zero_single_proposal_prompt_use_reflection_en = '''
Your task is to give the correct next step, given a science problem, an existing partial solution (not a complete answer) and some analysis for the next step.
Assuming the input is n-steps, then the format of the input is:
"Problem: ...
Existing Steps:
Step 1: ...
Step 2: ...
...
Step n: ...
Analysis: ..."

where ... denotes omitted input information.
If no existing steps are provided, you need to output the first step referring to the given analysis. Otherwise, you need to output the next step (step n+1) that you think is correct, following the ideas of the existing steps and provided analysis.
The output format is limited to:
"Next step: ..."
where ... indicates omitted output information, which is the part you should fill in. Your output should be a complete reasoning step that includes calculations, reasoning, choosing answers, etc.
Here is the input, please follow the restricted output format.

Problem: '''


zero_single_proposal_prompt_use_reflection_gpt_en = '''
Given a science problem, you need to answer the problem based on your existing knowledge. The input may include some existing steps for the solution and analysis for the next step, please give the next step of the solution specifically based on these information.

If no existing steps are provided, you need to refer to the analysis for the solution to give the first step in solving or calculating the question. If partial solution steps are provided, you need to output the next step of the answer following the ideas of the already existing steps and the provided analysis. If no analysis is given in the input, just output the next step following the idea of the existing steps. If the hint is not helpful or duplicates an existing step, then ignore it and output the next step.
The output format is limited to:
"Next step: ..."
where ... denotes omitted output information, which is what you should fill in to answer the next step. Your output should be a complete reasoning step, including calculations, reasoning, choosing answers, etc.
Here is the input, please follow the specified format for your output.

Problem: '''

single_reflection_prompt_en = '''
Given a science problem with existing answer steps (not necessarily complete answers), your task is to determine if the existing steps have solved the problem. If it has not been solved, give comments and brief ideas for next steps in response to the steps already in place.
Assuming that the steps already available are n steps, the input would be of the form:
"Problem: ...
Existing Steps:
Step 1: ...
Step 2: ...
...
Step n: ..."

where ... denotes omitted input information.
You need to distinguish between two cases and give the corresponding output.
Case 1: If these steps have already solved the problem and computed the final answer, then just output: "Problem solved" and nothing else.
Case 2: If the problem has not been completely solved, you need to analyze the existing steps, and point out the brief idea of the next step. If no existing steps are provided, then you need to briefly analyze the problem. The output format is limited to: "Analysis: ...", where ... indicates omitted output information, which is the part you should fill in.
Here is the input, please follow the requested output instructions, do not try to answer the whole question.

Problem: '''

single_reflection_prompt_simple_en = '''
You are an expert in science. Given a science problem and some corresponding steps (not necessarily complete) to answer it, you need to determine whether the given steps have completely solved the problem.

You need to distinguish between two cases and give the corresponding output.
Case 1: If the given steps have already solved the problem and provided the final answer to the question, then you should output: "Problem solved" and nothing else.
Case 2: If the given steps have not yet calculated the answer to the question or have not finished reasoning, then please output: "Problem unsolved" with no other content.
Note that if the existing steps do not compute the answer or do not simplify the answer expression as required by the question, then it should be considered unsolved.
Here is the input, please follow the requested output instructions, you do not need to answer the question.

Problem: '''

single_reflection_prompt_simple_mistral = '''
Given a science problem and some corresponding steps, if the given steps have already solved the problem and provided the final answer to the question, then you should output: "solved". Otherwise, please output: "unsolved".
Following the instruction, output "unsolved" or "solved", with no other information.

Problem: '''

self_critic_prompt = '''
Given a science problem and an existing solution, your task is to evaluate the correctness of the solution and provide an evaluation score. 
Your output should be a decimal ranging from 0 to 1. The more correct the solution is, the higher your evaluation score should be.

Problem:'''