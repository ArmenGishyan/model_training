from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# Paths
QLORA_DIR = "./qlora_out"           # Your QLoRA adapter folder
MERGED_DIR = "./merged_fp16_model"  # Where merged model will be saved

print("🔹 Loading QLoRA model with adapters...")
model = AutoPeftModelForCausalLM.from_pretrained(
    QLORA_DIR,
    torch_dtype="float16",  # change to torch.float32 if you want FP32
    device_map="cuda"
)

tokenizer = AutoTokenizer.from_pretrained("Llama-3.1-8B-Instruct")

torch.cuda.empty_cache()

print("🔹 Merging LoRA adapters into base model...")
model = model.merge_and_unload()

print(f"✅ Saving merged FP16 model to {MERGED_DIR}")
model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
