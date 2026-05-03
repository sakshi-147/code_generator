import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model once
@st.cache_resource
def load_model():
    model_name = "Salesforce/codegen-350M-mono"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# UI
st.title("💻 AI Code Synthesizer")
st.write("Generate Python code with explanation")

user_input = st.text_area("Describe the code you need:")

if st.button("🚀 Generate Code"):

    if user_input.strip() == "":
        st.warning("Please enter something")
    else:
        prompt = f"Write a simple Python function to {user_input}\n\nExplanation:"

        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = model.generate(
            inputs["input_ids"],
            max_length=200,
            do_sample=True,
            temperature=0.5
        )

        result = tokenizer.decode(tokens[0], skip_special_tokens=True)

        st.subheader("📄 Generated Output")
        st.code(result, language="python")