import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# LOAD MODEL (cached)
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "Salesforce/codegen-350M-mono"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="AI Code Generator", layout="centered")

st.title("💻 AI Code Generator & Explainer")
st.write("Generate code in multiple languages with explanation.")

# Language selection
language = st.selectbox(
    "Select Programming Language",
    ["Python", "Java", "C++", "JavaScript"]
)

# User input
user_input = st.text_area("Enter your problem statement")

# -------------------------------
# GENERATE CODE
# -------------------------------
def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    tokens = model.generate(
        inputs["input_ids"],
        max_length=250,
        do_sample=True,
        temperature=0.4,
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return result

# -------------------------------
# CLEAN OUTPUT
# -------------------------------
def clean_code(text):
    # Remove repeated prompts
    if "Explanation:" in text:
        text = text.split("Explanation:")[0]

    # Remove duplicate lines
    lines = list(dict.fromkeys(text.split("\n")))
    return "\n".join(lines).strip()

# -------------------------------
# SIMPLE EXPLANATION FUNCTION
# -------------------------------
def explain_code(code, language):
    explanation = "🔍 Line-by-line Explanation:\n\n"

    lines = code.split("\n")
    step = 1

    for line in lines:
        line = line.strip()

        if not line or line.startswith("//") or line.startswith("#"):
            continue

        explanation += f"{step}. `{line}` → "

        if "input" in line.lower():
            explanation += "Takes input from user.\n"
        elif "=" in line:
            explanation += "Stores value in a variable.\n"
        elif "for" in line or "while" in line:
            explanation += "Loop used to repeat steps.\n"
        elif "print" in line or "cout" in line or "console.log" in line:
            explanation += "Displays output.\n"
        elif "return" in line:
            explanation += "Returns result from function.\n"
        elif "*" in line:
            explanation += "Performs multiplication.\n"
        else:
            explanation += "Executes a step in program.\n"

        step += 1

    explanation += "\n📌 Overall:\nThis program takes input, processes it, and prints the result."

    return explanation

# -------------------------------
# BUTTON ACTION
# -------------------------------
if st.button("🚀 Generate Code"):

    if user_input.strip() == "":
        st.warning("Please enter a problem statement.")
    else:
        # Prompt
        prompt = f"""
Write only a clean {language} program to {user_input}.
Do not add explanation outside code.
Add comments inside code.
"""

        raw_output = generate_code(prompt)
        final_code = clean_code(raw_output)

        st.success("Code Generated Successfully!")

        # Show Code
        st.subheader("🔹 Generated Code")
        st.code(final_code, language=language.lower())

        # Explanation
        st.subheader("📘 Explanation")
        explanation = explain_code(final_code, language)
        st.text(explanation)

        # Download
        st.download_button(
            label="📥 Download Code",
            data=final_code,
            file_name="generated_code.txt",
            mime="text/plain"
        )
