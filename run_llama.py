from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

MODEL_ID = "./Llama-3.1-8B-Instruct"

print("Loading model... this may take a few minutes on first run.")

# Quantization config (to fit in 16GB T4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda"
)

print("✅ Model loaded! Ready to chat.\n")

# Interactive loop
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break

        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt").to("cuda")

        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

        # Decode & print
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"LLaMA: {response}\n")

    except KeyboardInterrupt:
        print("\n👋 Exiting chat.")
        break
