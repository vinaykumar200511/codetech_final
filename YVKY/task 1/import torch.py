import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text, max_length=150):
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        input_ids,
        max_length=max_length,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    print("Text Summarization Tool")
    print("Enter your article below:\n")
    article = input()
    result = summarize_text(article)
    print("\nSummary:\n")
    print(result)