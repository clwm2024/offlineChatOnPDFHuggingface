import os
import fitz  # PyMuPDF
from transformers import Trainer, TrainingArguments, AutoModelForQuestionAnswering, AutoTokenizer
from datasets import Dataset
import re

# 1. PDF-Inhalt aus allen PDF-Dateien im Verzeichnis extrahieren
def extract_text_from_pdfs(directory_path):
    all_text = ""
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                all_text += doc[page_num].get_text("text")
    return all_text

pdf_directory = "data/german_docs"
text = extract_text_from_pdfs(pdf_directory)

# 2. Text in Frage-Antwort-Paare aufteilen
def split_text_to_qa_pairs(text):
    # Hier ein einfaches Beispiel für Trennungen – passe es an dein PDF-Format an
    questions = re.findall(r"(?<=\bFrage\b:)(.*?)(?=\nAntwort:)", text, re.DOTALL)
    answers = re.findall(r"(?<=\bAntwort\b:)(.*?)(?=\nFrage:|\Z)", text, re.DOTALL)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in zip(questions, answers)]

qa_pairs = split_text_to_qa_pairs(text)
if not qa_pairs:
    raise ValueError("Keine Frage-Antwort-Paare gefunden. Stellen Sie sicher, dass die PDFs das richtige Format haben.")

# Datensatz erstellen
dataset = Dataset.from_list(qa_pairs)
dataset = dataset.train_test_split(test_size=0.2)

# 3. Modell und Tokenizer laden
model_name = "distilbert-base-german-cased"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenizer und Modell an den Text anpassen
def preprocess_data(examples):
    inputs = tokenizer(
        examples["question"],
        examples["answer"],
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    inputs["labels"] = inputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)

# 4. Training konfigurieren
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

# Trainer initialisieren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Training
trainer.train()

# 5. Modell speichern
model_save_path = ".models/sistilbertbasegermancased_trained"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Modell wurde unter {model_save_path} gespeichert.")