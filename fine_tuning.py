import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import json
import numpy as np
from tqdm import tqdm
from transformers import DataCollatorWithPadding
from accelerate import Accelerator


# in the fine-tuned, following the format in instruction fine-tuning

class SentenceScoreDataset(Dataset):
    def __init__(self, sentences, scores, tokenizer, max_length):
        self.sentences = sentences
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.sentences[idx],
            max_length=self.max_length,
            padding=False,  # No padding here
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.scores[idx], dtype=torch.float),
        }

# Replace the output layer with an MLP
class LlamaWithMLP(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model
        self.hidden_size = model.config.hidden_size
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size // 2, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last hidden state
        pooled_output = hidden_states.mean(dim=1)  # Mean pooling
        score = self.mlp_head(pooled_output).squeeze(-1)
        return score

# Initialize model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' # to prevent errors with FA
tokenizer.truncation_side = 'left' # to prevent cutting off last generation
device = torch.device("cuda:3")
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)


# Set up LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Type of task
    inference_mode=False,          # Training mode
    r=8,                           # LoRA rank
    lora_alpha=16,                 # Scaling factor
    lora_dropout=0.1,              # Dropout rate
)

# Apply LoRA to the model
base_model = get_peft_model(base_model, peft_config)
model = LlamaWithMLP(base_model)

# Prepare dataset
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

step1 = load_jsonl('/ssdscratch/byuan48/Reasoning/step1.jsonl')
sentences = [step1[i]['original_prompt'] + step1[i]['generated_text'] for i in range(len(step1))]
accuracies = np.load('/ssdscratch/byuan48/Reasoning/accuracies.npy')


dataset = SentenceScoreDataset(sentences, accuracies.flatten(order='C'), tokenizer, max_length=1024)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator,shuffle=True)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.MSELoss()
model.to(device)





# accelerator = Accelerator()
# dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)

# Number of epochs
epochs = 1

for epoch in range(epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0  # Track total loss for the epoch

    print(f"Starting Epoch {epoch + 1}/{epochs}...")

    # Wrap the dataloader with tqdm for a progress bar
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Zero gradients, forward pass, compute loss, and backprop
        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        # Update epoch loss
        epoch_loss += loss.item()

    # Print epoch summary
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_loss:.4f}")

# Save the model
torch.save({
    'base_model_state_dict': model.base_model.state_dict(),
    'mlp_head_state_dict': model.mlp_head.state_dict()
}, "./llama3_lora_mlp/llama_with_mlp.pth")

tokenizer.save_pretrained("./llama3_lora_mlp")

print("Fine-tuning completed and model saved!")
