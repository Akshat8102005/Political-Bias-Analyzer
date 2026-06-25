import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


test_df = pd.read_csv("../data/processed/test.csv")

test_dataset = Dataset.from_pandas(test_df)


tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


test_dataset = test_dataset.map(tokenize, batched=True)

test_dataset = test_dataset.rename_column("label", "labels")

test_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_SAVE_PATH
)

model.to(device)

trainer = Trainer(model=model)

predictions = trainer.predict(test_dataset)

preds = np.argmax(predictions.predictions, axis=1)

labels = predictions.label_ids


print("\nAccuracy :", accuracy_score(labels, preds))

print("Precision :", precision_score(labels, preds, average="weighted"))

print("Recall :", recall_score(labels, preds, average="weighted"))

print("F1 Score :", f1_score(labels, preds, average="weighted"))

print("\nClassification Report\n")

print(classification_report(labels, preds))


cm = confusion_matrix(labels, preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Right",
        "Lean Right",
        "Center",
        "Lean Left",
        "Left"
    ]
)

plt.figure(figsize=(8,6))

disp.plot(cmap="Blues")

os.makedirs("../reports", exist_ok=True)

plt.savefig(
    "../reports/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved!")
