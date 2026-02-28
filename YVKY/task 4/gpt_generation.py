from transformers import pipeline

# Load GPT model (GPT-2)
generator = pipeline("text-generation", model="gpt2")

# User prompt
prompt = "The importance of renewable energy in modern society"

# Generate text
output = generator(prompt, max_length=150, num_return_sequences=1)

print("Generated Text:\n")
print(output[0]['generated_text'])
