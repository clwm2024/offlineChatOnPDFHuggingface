from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import os
import shutil
import streamlit as st
import torch
from transformers import ElectraTokenizer, ElectraForQuestionAnswering

PROMPT_TEMPLATE = """
Beantworte die Frage im folgenden Kontext zur Projektmanagementmethode HERMES der Schweizer Bundesverwaltung:

{context}

---

Beantworte die Frage anhand des obenstehenden Kontexts zur Projektmanagementmethode HERMES der Schweizer Bundesverwaltung: {question}
"""

# Lade die Umgebungsvariablen
load_dotenv("settings.env")

# OpenAI API Key laden
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

CHROMA_PATH = "chroma"
# Angepasster Pfad zu deinen deutschen Dokumenten
DATA_PATH = "data/german_docs"

# Funktion zur Datenbankerstellung (nur wenn sie noch nicht existiert)
def generate_data_store():
    # Überprüfe, ob die Datenbank bereits existiert
    if os.path.exists(CHROMA_PATH):
        print("Datenbank existiert bereits. Lade vorhandene Datenbank.")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    else:
        print("Erstelle eine neue Datenbank.")
        documents = load_documents()
        chunks = split_text(documents)
        db = save_to_chroma(chunks)
    
    return db

# Dokumente laden
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    #loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents

# Text splitten
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Chroma-Datenbank speichern
def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    return db

def check_text_start(text,text2check):
    for txt in text2check:
        if text.startswith(txt):
            return True
            break
    return False

# Sample chat query function
def get_response_from_openai(prompt):
    # Placeholder for OpenAI API call (assuming API key setup)
    # response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=50)
    # return response.choices[0].text
    model = ChatOpenAI()
    response_text = model.predict(prompt)
    return response_text

def get_response_from_local_model(prompt,context):
    # Placeholder for local model response generation
    # model = AutoModelForCausalLM.from_pretrained("path_to_local_model")
    # tokenizer = AutoTokenizer.from_pretrained("path_to_local_model")
    # inputs = tokenizer(prompt, return_tensors="pt")
    # output = model.generate(inputs["input_ids"], max_length=50)
    # return tokenizer.decode(output[0], skip_special_tokens=True)
    # return "NOCH NICHT IMPLEMENTIERT"
    # Lade den Tokenizer und das Modell
    tokenizer = ElectraTokenizer.from_pretrained("Sahajtomar/German-question-answer-Electra")
    model = ElectraForQuestionAnswering.from_pretrained("Sahajtomar/German-question-answer-Electra")

    # Beispiel-Frage und -Kontext
    # question = "Wie hiess früher die Hauptstadt von Deutschland?"
    # question = "welche Stadt war vor Berlin Deutschlands Hauptstadt?"
    # context = "Berlin ist heute die Hauptstadt von Deutschland. Früher war Bonn die Hauptstadt und einst gab es sogar weitere Hauptstädte."
    # context = "Deutschland hat Bundesländer und ist föderal organisiert. Auch diverse Kreisstädte und Länderhauptstädte sowie Berlin als Hauptstadt von ganz Deutschland sind vorhanden. Bis Ende der 80er Jahre des letzten Jahrhunderts war Bonn die Hauptstadt von Westdeutschland."
    # Tokenisierung der Eingaben
    inputs = tokenizer(prompt, context, return_tensors='pt')

    # Vorhersage
    with torch.no_grad():
        outputs = model(**inputs)

    # Extrahiere die Start- und Endlogits
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Finde die Indizes für die Antwort
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1  # Inkludiere das End-Token

    # Hole die Antwort-Token
    answer_tokens = inputs['input_ids'][0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens)
    return answer


# Hauptfunktion für die App
def main():
    # Logo hinzufügen
    st.image("swisslogo.png", width=300)  # Passe den Pfad und die Größe an
    st.title("HERMES")
    #st.write(openai_api_key)
    st.write("welche Rollen kennt HERMES?")

    # Define options
    model_options = ["OpenAI Abfrage", "Lokales Modell"]

    # Radio button to select model
    selected_model = st.radio("Auswahl für die Modellabfrage:", model_options)
    # Placeholder for the chat response
    st.write("Gewählte Option:", selected_model)


    # Datenbank nur einmal erstellen oder laden
    db = generate_data_store()
    
    # Eingabe für die Frage
    query = st.text_input("Gib deine Frage ein:")

    if st.button("Frage beantworten"):
        if query:  # Überprüfe, ob eine Frage eingegeben wurde
            with st.spinner("Verarbeite deine Anfrage..."):
                # Suche ähnliche Dokumente in der bestehenden Chroma-Datenbank

                # neue verbesserte Version
                # Search the DB.
                results = db.similarity_search_with_relevance_scores(query, k=3)
                if len(results) == 0 or results[0][1] < 0.7:
                    print(f"Unable to find matching results.")
                    #aiAnswer = f"Tut mir leid, aber ich kann die Frage '{query}' nicht beantworten.\nÄndere bitte die Frage und versuche es erneut."
                    aiAnswer = f"Tut mir leid, aber ich kann die Frage '{query}' nicht beantworten.\nÄndere bitte die Frage und versuche es erneut.\nOder frage Bill Gates via 'billgeitsodergeitsnid@microsoft.com'"
                    st.write(aiAnswer)
                    return
                else:

                    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                    prompt = prompt_template.format(context=context_text, question=query)
                    print(prompt)

                    ##### MODELLWAHL
                    #model = ChatOpenAI()
                    #response_text = model.predict(prompt)
                    if selected_model == "OpenAI Abfrage":
                        response_text = get_response_from_openai(prompt)
                    else:
                        context = 'Beantworte die Frage anhand des obenstehenden Kontexts zur Projektmanagementmethode HERMES der Schweizer Bundesverwaltung.'
                        response_text = get_response_from_local_model(prompt,context)

                    print(f"********* {response_text} ***********")
                    sources = [doc.metadata.get("source", None) for doc, _score in results]
                    # Ersetze den Textteil in jedem Element der Liste
                    sources = [item.replace(DATA_PATH, '') for item in sources]
                    sources = [item.replace('/', '') for item in sources]

                    noMatchText = []
                    noMatchText.append("Sorry, I cannot provide an answer to the question.")
                    noMatchText.append("Unable to find matching results.")
                    if check_text_start(response_text,noMatchText):
                        aiAnswer = f"Tut mir leid, aber ich kann die Frage '{query}' nicht beantworten.\nÄndere bitte die Frage und versuche es erneut."
                    else:
                        aiAnswer = f"{response_text}\n\nQuellen: {sources}"

                    finalResponse = f"{aiAnswer}"
                    st.write(finalResponse)
                    return
        else:
            st.write("Bitte gib eine Frage ein.")

if __name__ == "__main__":
    main()