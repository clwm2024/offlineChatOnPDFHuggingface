import openai
import os
from dotenv import load_dotenv
import json

# Lade die Umgebungsvariablen aus der settings.env-Datei
load_dotenv("settings.env")

# Setze deinen OpenAI API Schlüssel
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_questions(prompt, n=10):
    # Verwende die ChatCompletion API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        n=n
    )
    
    # Extrahiere die Fragen aus der Antwort
    questions = [choice['message']['content'].strip() for choice in response['choices']]
    return questions

def generate_context(question):
    prompt = f"Erstelle einen Kontext zur Frage: '{question}' im Zusammenhang mit der HERMES-Projektmethode."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        n=1
    )
    context = response['choices'][0]['message']['content'].strip()
    return context

def create_json_format(questions):
    json_data = []
    for question in questions:
        context = generate_context(question)
        answer = f"Die Antwort auf '{question}' ist: [Antwort hier einfügen]"  # Platzhalter für die Antwort
        json_data.append({
            "context": context,
            "question": question,
            "answer": answer
        })
    return json_data

def save_to_json(data, output_file):
    # Speichern der Daten in einer JSON-Datei
    with open(output_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    n = 500  # Gesamtanzahl der Fragen, die generiert werden sollen
    all_json_data = []

    for i in range(n):
        print(f"Verarbeite die {i + 1}-te Frage...")
        prompt = f"Erstelle eine Frage zur Projektmethode HERMES."
        questions = generate_questions(prompt, n=1)  # Generiere eine Frage
        
        # Erstellen des JSON-Formats
        json_data = create_json_format(questions)
        all_json_data.extend(json_data)  # Hinzufügen der neuen Einträge

        # Zwischenspeichern nach 10 Einträgen
        if len(all_json_data) % 10 == 0:
            save_to_json(all_json_data, output_file='questions_answers.json')  # In derselben Datei speichern
            print(f"Die Fragen und Antworten wurden in 'questions_answers.json' gespeichert.")
    
    # Speichern der restlichen Einträge, falls vorhanden
    if len(all_json_data) % 10 != 0:
        save_to_json(all_json_data, output_file='questions_answers.json')  # Auch hier in derselben Datei speichern
        print("Die verbleibenden Fragen und Antworten wurden in 'questions_answers.json' gespeichert.")

if __name__ == "__main__":
    main()