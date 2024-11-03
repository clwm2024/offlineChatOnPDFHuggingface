from datasets import Dataset  # , load_metric
import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time
import torch
import transformers
'''
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments
from transformers import Trainer
'''
transformers.logging.set_verbosity_error()  # Reduziert die Log-Ausgabe

# Funktion zur Berechnung der Metriken
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Schritt 1: Lade die JSON-Daten und bereite sie für das Training vor
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Strukturieren der Daten für das Dataset-Format
    contexts = [entry["context"] for entry in data]
    questions = [entry["question"] for entry in data]
    answers = [{"text": entry["answer"], "answer_start": entry["context"].find(entry["answer"])} for entry in data]
    return contexts, questions, answers

# Schritt 4: Tokenisieren der Daten
def preprocess_function(examples, tokenizer):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = tokenized_examples.pop("offset_mapping")
    sample_map = tokenized_examples.pop("overflow_to_sample_mapping")
    answers = examples["answers"]

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Finde den Token-Start- und End-Index
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(tokenized_examples["input_ids"][i]) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)
        else:
            start_positions.append(0)
            end_positions.append(0)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

def main():

    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nAktuelle Zeit (Start Prozess):", current_time)

    step = 0
    step += 1
    print(f"Schritt #{step}")
    # Schritt 1: Lade die JSON-Daten und bereite sie für das Training vor
    workingdir = os.path.dirname(os.path.abspath(__file__))
    qadoc = os.path.join(workingdir,"questions_answers.json")
    contexts, questions, answers = load_data(qadoc)

    # Schritt 2: Erstellen eines Hugging Face Datasets und Daten aufteilen
    step += 1
    print(f"Schritt #{step}")
    dataset = Dataset.from_dict({
        "context": contexts,
        "question": questions,
        "answers": answers
    })
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Schritt 3: Modell und Tokenizer laden
    step += 1
    print(f"Schritt #{step}")
    model_name = "Sahajtomar/German-question-answer-Electra"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Schritt 4: Tokenisieren der Daten
    step += 1
    print(f"Schritt #{step}")
    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=eval_dataset.column_names)

    # Schritt 5: Trainingseinstellungen definieren
    step += 1
    print(f"Schritt #{step}")
    training_args = transformers.TrainingArguments(
        output_dir="models/trainedGermanQAElectra",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Schritt 6: Trainer erstellen
    step += 1
    print(f"Schritt #{step}")
    # Definiere die TrainingArguments wie im vorherigen Code beschrieben
    '''  
    training_args = transformers.TrainingArguments(     
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        fp16=True
    '''
    training_args = transformers.TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",  # Aktualisierte Einstellung
        fp16=False,  # Optional: Mixed Precision deaktivieren
    )

    # DataCollatorWithPadding für Tokenizer
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    # Trainer mit neuen Parametern
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,  # Aktualisierte Einstellung
    )

    '''
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )


    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
    )

    '''

    # Schritt 7: Training starten
    step += 1
    print(f"Schritt #{step} - Modell trainieren.")
    # Trainingsdauer messen
    start_time = time.time()  # Startzeitpunkt
    
    trainer.train()
    
    end_time = time.time()  # Endzeitpunkt
    training_duration = end_time - start_time  # Berechne die Dauer
    # Dauer ausgeben (in Sekunden)
    print(f"Training completed in {training_duration:.2f} seconds.")

    # Modell speichern
    print("Modell speichern....!")
    trainer.save_model("models/trainedGermanQAElectra")
    print("Modell abgespeichert!")

    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"***************************\nAktuelle Zeit (Ende Prozess):", current_time)


if __name__ == "__main__":
    main()