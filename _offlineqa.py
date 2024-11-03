from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.schema import Document  # Importiere die Document-Klasse

import torch


results = [(Document(metadata={'source': 'data/german_docs/HERMES 2022 _ REFERENZHANDBUCH – Projektmanagement.pdf', 'start_index': 192444}, page_content='HERMES-spezifisch'), 0.8476647363068763), (Document(metadata={'source': 'data/german_docs/HERMES 2022 _ REFERENZHANDBUCH – Projektmanagement.pdf', 'start_index': 207982}, page_content='HERMES-spezifisch'), 0.8476647363068763), (Document(metadata={'source': 'data/german_docs/HERMES 2022 _ REFERENZHANDBUCH – Projektmanagement.pdf', 'start_index': 258872}, page_content='HERMES-spezifisch'), 0.8476647363068763)]
#results_string = ', '.join(results)
#results = results_string.split(', ')

    
# Lade das lokal gespeicherte Modell und Tokenizer
# local_model_path = "./local_german_gpt2"
local_model_path = "/Users/hansjoerg.stark/development/Python/chatgpt-HERMES/.models/distilbertbasegermancased_trained"
#local_model_path = ".models/distilbertbasegermancased_trained"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# Der Kontext wird aus den Suchergebnissen erstellt
context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

# Erstelle die Eingabeaufforderung für das Modell
query_text = 'Wo kommt HERMES Projektmanagement der Bundesverwaltung in der Schweiz zum Einsatz?'
input_text = f"Kontext:\n{context_text}\n\nFrage: {query_text}\nAntwort:"

# Tokenisiere die Eingabeaufforderung
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generiere die Antwort
with torch.no_grad():
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)

# Dekodiere die Antwort
response_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Extrahiere die Antwort
formatted_response = response_text.split("Antwort:")[-1].strip()  # Die Antwort nach "Antwort:" extrahieren
sources = [doc.metadata.get("source", None) for doc, _score in results]

print(f"Response: {formatted_response}\nSources: {sources}")