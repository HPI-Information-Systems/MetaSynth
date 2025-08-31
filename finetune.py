import sys
import typing
sys.modules['typing'].Any = typing.Any

import builtins
builtins.Any = typing.Any
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

from datasets import Dataset
from trl import SFTConfig, SFTTrainer, clone_chat_template
import os 
import argparse
from transformers import AutoTokenizer
import time

os.environ["WANDB_PROJECT"] = "Master Thesis"

parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="Beta Value for DPO", default=0.1)
parser.add_argument("--llm_path", type=str, help="Path to the LLM that should be finetuned")
parser.add_argument("--train_ds", type=str, help="Path to the training dataset")
parser.add_argument("--eval_ds", type=str, help="Path to the evaluation dataset")
parser.add_argument("--output_dir", type=str, help="Directory to save the checkpoints")
parser.add_argument("--learning_rate", type=float, help="Learning rate for training", default=5e-5)
parser.add_argument("--epochs", type=int, help="Number of training epochs", default=1)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

model, tokenizer = FastLanguageModel.from_pretrained(args.llm_path, max_seq_length = 2**14)
tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
model = FastLanguageModel.get_peft_model(model)

def formatting_func(examples):
    return tokenizer.apply_chat_template(examples["messages"], tokenize=False, add_generation_prompt=True)

raw_train_dataset = Dataset.load_from_disk(args.train_ds).shuffle(seed=42)

train_dataset = Dataset.from_dict({"text": [tokenizer.apply_chat_template(ex, tokenize=False, add_generation_prompt=True) for ex in raw_train_dataset["messages"]]})

raw_eval_dataset = Dataset.load_from_disk(args.eval_ds).shuffle(seed=42)

eval_dataset = Dataset.from_dict({"text": [tokenizer.apply_chat_template(ex, tokenize=False, add_generation_prompt=True) for ex in raw_eval_dataset["messages"]]})

training_args = SFTConfig(output_dir=args.output_dir, 
                            logging_steps=10, 
                            bf16=True, 
                            num_train_epochs=args.epochs, 
                            eval_strategy="steps", 
                            eval_steps=5000, 
                            save_strategy="epoch", 
                            packing=False, 
                            max_seq_length=10000, 
                            per_device_train_batch_size=3, 
                            per_device_eval_batch_size=3,
                            learning_rate=args.learning_rate,
                            run_name=f"{args.llm_path.split('/')[-1]}_{args.train_ds.split("/")[-2]}_{args.epochs}_SFT",
                            )   
trainer = SFTTrainer(model=model, args=training_args, tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset, dataset_text_field="text")
start_time = time.time()
trainer.train()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")
os.makedirs(f"{args.output_dir}/model", exist_ok=True)
model.save_pretrained_merged(f"{args.output_dir}/model", tokenizer, save_method = "merged_16bit",)