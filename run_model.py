import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import os

torch.cuda.empty_cache()

# 1. Define the local path to the model's directory
local_model_path = "./Llama-3.1-8B-Instruct"

# Check if the model directory exists
if not os.path.isdir(local_model_path):
    print(f"Error: Model directory not found at '{local_model_path}'.")
    print("Please make sure the directory exists and contains the model files.")
    exit()

# 2. Load the tokenizer from the local directory.
tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False)

# 3. Load the model from the local directory.
# `device_map="auto"` is recommended for best performance on GPU.
print("Loading model from local machine...")
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
print("Model loaded successfully.")

# 4. Prepare the chat history using the Llama 3.1 instruct format.
messages = [
    {"role": "system", "content": "You are a helpful and friendly assistant."},
    {"role": "user", "content": "What is the capital of Canada?"},
]

# 5. Tokenize the messages.
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# 6. Generate the model's response.
print("\nGenerating response...")
streamer = TextStreamer(tokenizer)
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    streamer=streamer
)

# 7. Decode and print the final full response (already printed by the streamer).
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Full Final Response ---")
print(full_response)

