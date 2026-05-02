from openai import OpenAI
from dotenv import load_dotenv
import os

# 1. This loads the key from your .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. This function sends your request to OpenAI
def generate_code(prompt):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=200,
        temperature=0.2,
    )
    return response.choices[0].text.strip()

# 3. Test it out!
if __name__ == "__main__":
    test_prompt = "Write a Python function to add two numbers."
    print("Sending request to OpenAI...")
    code = generate_code(test_prompt)
    print("\n--- GENERATED CODE ---")
    print(code)