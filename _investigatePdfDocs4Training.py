import os
import fitz  # PyMuPDF
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Schritt 1: PDF-Texte aus einem Verzeichnis extrahieren
def extract_text_from_pdfs(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(directory, filename)) as pdf:
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    texts.append(page.get_text())
    return texts

# PDF-Texte extrahieren
pdf_directory = "data/german_docs"
fullDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),pdf_directory)
pdf_texts = extract_text_from_pdfs(fullDir)

# Schritt 2: Datensatz für das Training erstellen
combined_text = "\n\n".join(pdf_texts)
dataset = Dataset.from_dict({"text": [combined_text]})

# Schritt 3: Modell und Tokenizer für MLM laden
model_name = "distilbert-base-german-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Textdaten tokenisieren
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Schritt 4: DataCollator für Masked Language Modeling hinzufügen
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Trainingseinstellungen konfigurieren
training_args = TrainingArguments(
    output_dir=".models/distilbertbasegermancased_trained",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer mit DataCollator und tokenisiertem Datensatz
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Modell trainieren
trainer.train()

# Modell und Tokenizer speichern
trainer.save_model(".models/distilbertbasegermancased_trained")
tokenizer.save_pretrained(".models/distilbertbasegermancased_trained")

print("Das Modell wurde erfolgreich trainiert und gespeichert.")