import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model (optimized for local use)
model_name = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_simple_code():
    user_input = input_field.get("1.0", tk.END).strip()
    # Adding a clear instruction for simple output
    prompt = f"Write a simple Python function to {user_input}\n\nExplanation:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = model.generate(inputs["input_ids"], max_length=200, do_sample=True, temperature=0.5)
    result = tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    output_field.delete("1.0", tk.END)
    output_field.insert(tk.END, result)

# Build GUI with a Professional Dark Theme
root = tk.Tk()
root.title("AI Code Generator - FY Project")
root.geometry("600x650")
root.configure(bg="#2d2d2d") # Dark background

# Styling
label_font = ("Helvetica", 12, "bold")
text_font = ("Consolas", 11)

tk.Label(root, text="Enter your coding task:", bg="#2d2d2d", fg="#ffffff", font=label_font).pack(pady=15)
input_field = tk.Text(root, height=4, width=60, font=text_font, bg="#3d3d3d", fg="#ffffff", insertbackground="white")
input_field.pack(pady=5)

tk.Button(root, text="GENERATE CODE", command=generate_simple_code, bg="#007acc", fg="white", font=("Helvetica", 10, "bold"), relief="flat", padx=20).pack(pady=15)

tk.Label(root, text="Generated Code & Explanation:", bg="#2d2d2d", fg="#ffffff", font=label_font).pack(pady=5)
output_field = scrolledtext.ScrolledText(root, height=18, width=70, font=("Consolas", 10), bg="#1e1e1e", fg="#00ff00", insertbackground="white")
output_field.pack(pady=10, padx=10)

root.mainloop()