from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

inputs = tokenizer("Hello from Codespaces!", return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0]))
