import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from config import *

print("Imports completed")

# ------------------------------
# Random Seeds
# ------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ------------------------------
# Load Dataset
# ------------------------------

print("Loading dataset...")

train_df = pd.read_csv("../data/processed/train.csv")
test_df = pd.read_csv("../data/processed/test.csv")

print(train_df.shape)
print(test_df.shape)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ------------------------------
# Tokenizer
# ------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )

print("Tokenizing train dataset...")

train_dataset = train_dataset.map(tokenize, batched=True)

print("Tokenizing test dataset...")

test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# ------------------------------
# Model
# ------------------------------

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

model.to(device)

print("Model loaded")

# ------------------------------
# Metrics
# ------------------------------

def compute_metrics(pred):

    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

print("Metrics ready")

# ------------------------------
# Training Arguments
# ------------------------------

training_args = TrainingArguments(
    output_dir="../models/checkpoints",

    overwrite_output_dir=True,

    num_train_epochs=EPOCHS,

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="epoch",

    load_best_model_at_end=True,

    save_total_limit=2,

    report_to="none"
)

print("TrainingArguments created")

# ------------------------------
# Trainer
# ------------------------------

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("Trainer created")

# ------------------------------
# Train
# ------------------------------

print("Starting Training...\n")

trainer.train()

print("Training finished")

# ------------------------------
# Evaluate
# ------------------------------

metrics = trainer.evaluate()

print(metrics)

# ------------------------------
# Save
# ------------------------------

trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("Model Saved Successfully!")