from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "dbmdz/german-gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Speichere das Modell lokal
model.save_pretrained("./local_german_gpt2")
tokenizer.save_pretrained("./local_german_gpt2")