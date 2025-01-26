from datasets import load_dataset
from reasoning.tools.utils import load_jsonl, load_model
import json
import random
import wandb
import torch

def check(task, evaluator, model_name, device):
    
    wandb.init(project="reasoning_evaluation", entity='yuanbo096')

    
    model, tokenizer = load_model(model_name,device)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset("lighteval/MATH", split=f"{task}", trust_remote_code=True)

    batch_size = 100

    total_correct = 0
    total_samples = 0

    for index in range(indices):
        
        whole_solutions = []
        whole_problems = []
        whole_expected_solutions = []
        
        print(f'processing the {index}-th batch' + '*'*50)
        sub_dataset = dataset.select(range(index * samples, (index + 1) * samples))
        expected_solutions = [sub_dataset[t]['solution'] for t in range(len(sub_dataset))]
        expected_solutions = [element for element in expected_solutions for _ in range(step1_batch_size)]
        whole_expected_solutions.extend(expected_solutions)

        step1 = load_jsonl(f"/ssdscratch/byuan48/Reasoning/generated_datasets/step1_large_{task}_{index}.jsonl")
        step2 = load_jsonl(f"/ssdscratch/byuan48/Reasoning/generated_datasets/step2_large_{task}_{index}.jsonl")

        for i in range(0, len(step1)):
            j = random.randint(0, step2_batch_size - 1)
            whole_problems.append(step1[i]['original_prompt'][171:])
            whole_solutions.append(step1[i]['generated_text'] + step2[i * step2_batch_size + j]['generated_text'])
    
        print('data loaded')
        evaluation_results = []
        for i in range(len(whole_problems)):
            # Generate answer using model
            input_ids = tokenizer.encode(whole_problems[i], return_tensors="pt").to(model.device)
            output = model.generate(input_ids, max_length=50, num_return_sequences=1)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # Evaluate the generated answer
            response = evaluator.evaluate(generated_text, whole_expected_solutions[i])
            print(f'evaluating the {i}-th sample')
            evaluation_results.append({'problem': whole_problems[i], 'solution': generated_text, 'expected_solution': whole_expected_solutions[i], 'evaluation': response})

            # Check if the solution is correct
            if response['correct']:
                total_correct += 1
            total_samples += 1

        # Log results to wandb
        wandb.log({"batch_index": index, "accuracy": total_correct / total_samples})

        with open(f'./data/evaluation_results_{task}.jsonl', 'a') as file:
            for result in evaluation_results:
                file.write(json.dumps(result) + '\n')

    # Final accuracy report
    final_accuracy = total_correct / total_samples
    print(f'Final accuracy: {final_accuracy}')
    wandb.log({"final_accuracy": final_accuracy})
    wandb.finish()

if __name__ == '__main__':
    evaluator = Evaluator()
    # task = 'test'; indices = 50
    task = 'train'; indices = 75
    check(task, indices, evaluator)
    print('evaluation completed')
    print('start analysis')
    print(evaluator.error_show())

