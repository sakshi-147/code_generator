from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Load the free model from Hugging Face
# This is a smaller, fast model perfect for FY projects.
model_name = "Salesforce/codegen-350M-mono"

print("Loading model... (this happens only once).")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_code_hf(prompt):
    # Prepare the input for the model
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate code
    # We set max_length=200 so it doesn't run forever
    tokens = model.generate(
        inputs["input_ids"], 
        max_length=200, 
        do_sample=True, 
        temperature=0.5
    )
    
    # Decode and return the result as text
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Test your new local generator
    user_prompt = "def reverse_string(s):"
    print("\nGenerating code locally...")
    result = generate_code_hf(user_prompt)
    print("\n--- GENERATED CODE ---")
    print(result)