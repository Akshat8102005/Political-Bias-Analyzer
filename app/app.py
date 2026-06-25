import torch
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import *

st.set_page_config(
    page_title="Political Bias Analyzer",
    page_icon="📰",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "roberta_bias_classifier"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()
model.eval()

label_map = {
    0: "Right",
    1: "Lean Right",
    2: "Center",
    3: "Lean Left",
    4: "Left"
}

st.title("📰 Political Bias Analyzer")

st.write(
    "Analyze the political leaning of a news article using a fine-tuned RoBERTa model."
)

text = st.text_area(
    "Paste an article below",
    height=250
)

if st.button("Analyze"):

    if len(text.strip()) == 0:
        st.warning("Please enter some text.")
        st.stop()

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

    prediction = probs.argmax()

    confidence = probs[prediction]

    st.success(
        f"Prediction: **{label_map[prediction]}**"
    )

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    st.subheader("Class Probabilities")

    for i in range(NUM_LABELS):

        st.progress(float(probs[i]))

        st.write(
            f"{label_map[i]} : {probs[i]*100:.2f}%"
        )
        
