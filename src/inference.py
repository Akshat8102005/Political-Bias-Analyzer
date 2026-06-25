import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {device}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_SAVE_PATH
)

model.to(device)
model.eval()

label_map = {
    0: "Right",
    1: "Lean Right",
    2: "Center",
    3: "Lean Left",
    4: "Left"
}


def probability_bar(prob):

    filled = int(prob * 20)

    return "█" * filled + "░" * (20 - filled)


while True:

    text = input("\nEnter article (type 'quit' to exit):\n\n")

    if text.lower() == "quit":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    probs = probabilities.cpu().numpy()

    sorted_indices = probs.argsort()[::-1]

    top_prediction = sorted_indices[0]
    second_prediction = sorted_indices[1]

    confidence = probs[top_prediction]
    second_confidence = probs[second_prediction]

    print("\n" + "=" * 60)

    print(f"Prediction : {label_map[top_prediction]}")
    print(f"Confidence : {confidence*100:.2f}%")

    print()

    print(
        f"Second Choice : {label_map[second_prediction]} "
        f"({second_confidence*100:.2f}%)"
    )

    if confidence - second_confidence < 0.10:
        print("\n⚠️  Mixed or uncertain political framing detected.")

    print("\nProbability Distribution\n")

    for idx in sorted_indices:

        print(
            f"{label_map[idx]:12s}"
            f"{probability_bar(probs[idx])}"
            f" {probs[idx]*100:.2f}%"
        )

    print("=" * 60)