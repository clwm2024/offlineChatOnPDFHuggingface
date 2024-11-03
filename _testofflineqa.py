
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import ElectraTokenizer, ElectraForQuestionAnswering
import torch

# Lade den Tokenizer und das Modell
tokenizer = ElectraTokenizer.from_pretrained("Sahajtomar/German-question-answer-Electra")
model = ElectraForQuestionAnswering.from_pretrained("Sahajtomar/German-question-answer-Electra")

'''
local_model_path = ".models/distilbertbasegermancased_trained"
#local_model_path = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(local_model_path)
#model = AutoModelForQuestionAnswering.from_pretrained(local_model_path)
#model = AutoModelForCausalLM.from_pretrained(local_model_path)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-german-cased')
model = DistilBertForQuestionAnswering.from_pretrained('.models/distilbertbasegermancased_trained')
'''


# Beispiel für die Verwendung des Modells zur Textgenerierung
input_text = "Was ist HErMES?"
context = "HERMES ist eine Methode für die Projektorganisation, -führung, -leitung und generelle Umsetzung und den Lebenszyklus von Projeten in der Budesverwaltung der Schweiz."

inputs = tokenizer.encode(input_text,context, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, num_return_sequences=1)

# Entschlüsseln der generierten Tokens
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generierte Antwort:", output_text)



'''
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Setze den korrekten, absoluten Pfad für das Modell
local_model_path = "/Users/hansjoerg.stark/development/Python/chatgpt-HERMES/.models/distilbertbasegermancased_trained"

# Lade das lokal gespeicherte Modell und den Tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForMaskedLM.from_pretrained(local_model_path)

# Beispiel für die Verwendung des Modells
input_text = "Hier ist eine unvollständige Antwort auf deine Frage: [MASK]"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)

# Analysiere die Ergebnisse
logits = outputs.logits
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
mask_token_logits = logits[0, mask_token_index, :]
top_5_tokens = mask_token_logits.topk(5, dim=1).indices[0].tolist()

# Zeige die Top 5 Vorhersagen
predicted_tokens = [tokenizer.decode([token]) for token in top_5_tokens]
print("Mögliche Maskenfüllungen:", predicted_tokens)
'''