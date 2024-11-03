from transformers import ElectraTokenizer, ElectraForQuestionAnswering, AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Lade den Tokenizer und das Modell
tokenizer = ElectraTokenizer.from_pretrained("Sahajtomar/German-question-answer-Electra")
model = ElectraForQuestionAnswering.from_pretrained("Sahajtomar/German-question-answer-Electra")
tokenizerloc = AutoTokenizer.from_pretrained("/Users/hansjoerg.stark/development/Python/chatgpt-HERMES/.models/distilbertbasegermancased_trained")
modelloc = AutoModelForQuestionAnswering.from_pretrained("/Users/hansjoerg.stark/development/Python/chatgpt-HERMES/.models/distilbertbasegermancased_trained")

# Beispiel-Frage und -Kontext
#question = "Wie hiess früher die Hauptstadt von Deutschland?"
question = "welche Stadt war vor Berlin Deutschlands Hauptstadt?"
#context = "Berlin ist heute die Hauptstadt von Deutschland. Früher war Bonn die Hauptstadt und einst gab es sogar weitere Hauptstädte."
context = "Deutschland hat Bundesländer und ist föderal organisiert. Auch diverse Kreisstädte und Länderhauptstädte sowie Berlin als Hauptstadt von ganz Deutschland sind vorhanden. Bis Ende der 80er Jahre des letzten Jahrhunderts war Bonn die Hauptstadt von Westdeutschland."
# Tokenisierung der Eingaben
#inputs = tokenizer(question, context, return_tensors='pt')

question2 = "Welche Rollen kennt HERMES?"
context2 = 'Beantworte die Frage anhand des obenstehenden Kontexts zur Projektmanagementmethode HERMES der Schweizer Bundesverwaltung.'
# inputs = tokenizer(question2, context2, return_tensors='pt')
inputs = tokenizerloc(question2, context2, return_tensors='pt')

# Vorhersage
with torch.no_grad():
    # outputs = model(**inputs)
    outputs = modelloc(**inputs)

# Extrahiere die Start- und Endlogits
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Finde die Indizes für die Antwort
start_index = torch.argmax(start_logits)
end_index = torch.argmax(end_logits) + 1  # Inkludiere das End-Token

# Hole die Antwort-Token
answer_tokens = inputs['input_ids'][0][start_index:end_index]
# answer = tokenizer.decode(answer_tokens)
answer = tokenizerloc.decode(answer_tokens)


print("Antwort:", answer)

