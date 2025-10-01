from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_DIR = "./merged_fp16_model"

print("🔹 Loading merged model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,  # change to torch.float32 if you merged as FP32
    device_map="auto"
)

print("✅ Model loaded! Ready for interactive chat.\n")

# Interactive loop
while True:
    prompt = input("📝 Enter your prompt (or type 'exit' to quit):\n> ").strip()
    if prompt.lower() in ["exit", "quit", "q"]:
        print("👋 Exiting chat.")
        break

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    # Decode & print
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n🔹 Model response:\n")
    print(response)
    print("\n" + "="*80 + "\n")
