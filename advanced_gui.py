import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading

# 1. Load model (only once)
model_name = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Logic to run generation without freezing the window
def run_generation():
    user_input = input_field.get("1.0", tk.END).strip()
    prompt = f"Write a simple Python function to {user_input}\n\nExplanation:"
    
    # Generate code
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = model.generate(inputs["input_ids"], max_length=250, do_sample=True, temperature=0.5)
    result = tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    output_field.delete("1.0", tk.END)
    output_field.insert(tk.END, result)

def generate_code_thread():
    output_field.delete("1.0", tk.END)
    output_field.insert(tk.END, "Analyzing prompt... Please wait...")
    threading.Thread(target=run_generation).start()

# 3. Build Advanced GUI
root = tk.Tk()
root.title("AI Code Synthesizer v1.0")
root.geometry("800x850") # Slightly larger window
root.configure(bg="#0f172a") # Deep Navy Background

# Styling
label_font = ("Segoe UI", 12, "bold")
text_font = ("Consolas", 11)

tk.Label(root, text="AI CODE SYNTHESIZER", bg="#0f172a", fg="#60a5fa", font=("Segoe UI", 24, "bold")).pack(pady=15)
tk.Label(root, text="Describe the code you need:", bg="#0f172a", fg="#e2e8f0", font=label_font).pack(pady=5)

input_field = tk.Text(root, height=4, width=80, font=("Segoe UI", 11), bg="#1e293b", fg="white", insertbackground="white", relief="flat", borderwidth=5)
input_field.pack(pady=5)

btn = tk.Button(root, text="GENERATE INTELLIGENT CODE", command=generate_code_thread, bg="#1e40af", fg="white", font=("Segoe UI", 12, "bold"), relief="flat", padx=20, pady=10)
btn.pack(pady=15)

tk.Label(root, text="Output Console:", bg="#0f172a", fg="#e2e8f0", font=label_font).pack(pady=5)
# INCREASED HEIGHT HERE from 18 to 28 for less scrolling
output_field = scrolledtext.ScrolledText(root, height=28, width=90, font=("Consolas", 11), bg="#020617", fg="#38bdf8", insertbackground="white", relief="flat")
output_field.pack(pady=10, padx=20)

root.mainloop()