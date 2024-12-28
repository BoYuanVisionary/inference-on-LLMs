import argparse
import json
import os
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm
from llama_mlp_model import LlamaWithMLP
from peft import get_peft_model, LoraConfig, TaskType
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with LoRA and MLP head.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name or path.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset (JSONL format).")
    parser.add_argument("--accuracies_path", type=str, required=True, help="Path to the accuracies (NumPy file).")
    parser.add_argument("--output_dir", type=str, default="./output", help="Base directory to save fine-tuned models.")
    parser.add_argument("--pooling_method", type=str, choices=["last_token", "mean_pool"], default="last_token", help="Pooling method for the hidden states.")
    parser.add_argument("--output_layer", type=str, choices=["mlp", "linear"], help="Output layer type (auto-decided if not specified).")
    parser.add_argument("--fine_tune_base", action="store_true", help="Whether to fine-tune the base model.")
    parser.add_argument("--wandb_project", type=str, default="llama_finetuning", help="WandB project name.")
    return parser.parse_args()


def generate_output_dir(base_dir, args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{args.pooling_method}_{args.output_layer}_{'fine_tuned' if args.fine_tune_base else 'frozen'}_{timestamp}"
    return os.path.join(base_dir, dir_name)


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
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.scores[idx], dtype=torch.float),
        }


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def main():
    args = parse_args()

    # Automatically decide output_layer if not specified
    if not args.output_layer:
        args.output_layer = "mlp" if args.pooling_method == "mean_pool" else "linear"

    # Initialize WandB
    wandb.init(project=args.wandb_project, config=vars(args))
    config = wandb.config

    # Generate output directory
    output_dir = generate_output_dir(config.output_dir, config)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Load dataset
    step1 = load_jsonl(config.dataset_path)
    sentences = [f"{entry['original_prompt']}{entry['generated_text']}" for entry in step1]
    accuracies = np.load(config.accuracies_path)

    dataset = SentenceScoreDataset(sentences, accuracies.flatten(order="C"), tokenizer, max_length=config.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=data_collator, shuffle=True)

    # Load base model and apply LoRA
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    base_model = get_peft_model(base_model, peft_config)

    # Define model with configuration
    model = LlamaWithMLP(base_model, config.pooling_method, config.output_layer, config.fine_tune_base)
    model.to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCELoss()

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        print(f"Starting Epoch {epoch + 1}/{config.epochs}...")

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        wandb.log({"epoch": epoch + 1, "average_loss": avg_loss})
        print(f"Epoch {epoch + 1}/{config.epochs} completed. Average Loss: {avg_loss:.4f}")

        # Save the model
        model_save_path = os.path.join(output_dir, f"llama_with_mlp_{epoch}.pth")
        torch.save({
            "base_model_state_dict": model.base_model.state_dict(),
            "mlp_head_state_dict": model.output_layer.state_dict(),
        }, model_save_path)
        print(f"Model saved to {model_save_path}")

    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuning completed and tokenizer saved to {output_dir}!")
    wandb.finish()


if __name__ == "__main__":
    main()
