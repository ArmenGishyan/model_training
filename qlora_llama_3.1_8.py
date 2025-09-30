import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# --------------------
# Config
# --------------------
MODEL_ID = "./Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./qlora_out"

# Load small dataset (for demo). Replace with your dataset.
# Example: use "imdb" dataset, only 2k samples to fit in T4.
dataset = load_dataset("json", data_files="trainig_data.json")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    # Combine Instruction + Response into one training text
    text = f"Instruction: {example['Instruction']}\nResponse: {example['Response']}"
    return tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# Apply formatting
tokenized_dataset = dataset["train"].map(tokenize_function, batched=False)

# Load base model in 4-bit for QLoRA
print("🔹 Loading base model in 4-bit...")
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

# Apply QLoRA (LoRA adapters on top of 4-bit model)
print("🔹 Applying LoRA adapters...")

lora_config = LoraConfig(
    r=64,                          # rank
    lora_alpha=16,                 # scaling
    target_modules=["q_proj","v_proj"],  # apply to attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Training args (small, T4-friendly)
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=False,
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    optim="adamw_torch"
)

# Use TRL's SFTTrainer (supervised fine-tuning)
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    args=training_args
)

print("🔹 Starting QLoRA training...")
trainer.train()

print(f"✅ Training finished! Adapters saved to {OUTPUT_DIR}")
model.save_pretrained("./qlora_out")


