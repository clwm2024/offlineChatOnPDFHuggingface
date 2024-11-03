from dotenv import load_dotenv
import json
import openai
import os

# Lade die Umgebungsvariablen
load_dotenv("settings.env")
# OpenAI API Key laden
openai_api_key = os.getenv("OPENAI_API_KEY")
# Setze deinen OpenAI API-Key
openai.api_key = openai_api_key



def generate_questions(prompt, n=500):
    questions = []
    print(n)
    
    while len(questions) < n:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None
        )

        # Extrahiere die Fragen aus der Antwort
        generated_text = response['choices'][0]['message']['content'].strip()
        questions_batch = generated_text.split('\n')

        # Füge die Fragen zur Liste hinzu, filtere leere Fragen
        for question in questions_batch:
            if question and question not in questions:
                questions.append(question)

    return questions

# Definiere das Prompt für die Generierung
prompt = "Bitte generiere 10 Fragen zur Projektmethode HERMES. Jede Frage sollte klar und präzise formuliert sein."

# Generiere die Fragen
questions = generate_questions(prompt,n=10)

# Speichere die Fragen in einer JSON-Datei
with open("hermes_questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=4)

print(f"Generierte {len(questions)} Fragen und gespeichert in 'hermes_questions.json'.")