
gui_code = """
import tkinter as tk
from tkinter import ttk, scrolledtext
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import threading

# Load model globally
model_name = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_code_thread():
    # Show status
    output_field.delete("1.0", tk.END)
    output_field.insert(tk.END, "Generating... please wait...")
    
    # Run generation in a thread so UI doesn't freeze
    threading.Thread(target=run_generation).start()

def run_generation():
    user_input = input_field.get("1.0", tk.END).strip()
    prompt = f"Write a simple Python function to {user_input}\\n\\nExplanation:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = model.generate(inputs["input_ids"], max_length=200, do_sample=True, temperature=0.5)
    result = tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    output_field.delete("1.0", tk.END)
    output_field.insert(tk.END, result)

# Build GUI with a Deep Blue Modern Theme
root = tk.Tk()
root.title("Advanced AI Code Synthesizer")
root.geometry("700x750")
root.configure(bg="#0f172a") # Deep Navy Background

# Styling
style = ttk.Style()
style.theme_use('clam')
style.configure("TLabel", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 12, "bold"))
style.configure("TButton", font=("Segoe UI", 11, "bold"), background="#1e40af", foreground="white")

tk.Label(root, text="AI Code Synthesizer", bg="#0f172a", fg="#60a5fa", font=("Segoe UI", 24, "bold")).pack(pady=20)
tk.Label(root, text="Describe the code you need:", bg="#0f172a", fg="#e2e8f0", font=("Segoe UI", 12)).pack(pady=5)

input_field = tk.Text(root, height=4, width=70, font=("Segoe UI", 11), bg="#1e293b", fg="white", insertbackground="white", relief="flat", borderwidth=5)
input_field.pack(pady=10)

btn = tk.Button(root, text="GENERATE INTELLIGENT CODE", command=generate_code_thread, bg="#1e40af", fg="white", font=("Segoe UI", 12, "bold"), relief="flat", padx=20, pady=10)
btn.pack(pady=20)

tk.Label(root, text="Output Console:", bg="#0f172a", fg="#e2e8f0", font=("Segoe UI", 12)).pack(pady=5)
output_field = scrolledtext.ScrolledText(root, height=18, width=80, font=("Consolas", 11), bg="#020617", fg="#38bdf8", insertbackground="white", relief="flat")
output_field.pack(pady=10, padx=20)

root.mainloop()
"""

with open("advanced_gui.py", "w") as f:
    f.write(gui_code)