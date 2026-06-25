# 📰 Political Bias Analyzer using RoBERTa

A transformer-based Natural Language Processing (NLP) application that classifies the political bias of news articles into five categories using a fine-tuned **RoBERTa** model.

The project compares a traditional machine learning baseline (TF-IDF + Logistic Regression) with a transformer-based approach and provides an interactive **Streamlit** application for real-time inference.

---

## Features

* Fine-tuned **RoBERTa** model for political bias classification
* Five-class prediction:

  * Right
  * Lean Right
  * Center
  * Lean Left
  * Left
* TF-IDF + Logistic Regression baseline
* Interactive Streamlit web application
* Probability distribution for all classes
* Confidence score for each prediction
* Model evaluation with confusion matrix and classification report

---

## Project Structure

```
Political-Bias-Analyzer/

│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   └── confusion_matrix.png
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train_transformer.py
│   ├── evaluate_transformer.py
│   └── inference.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The model is trained on a labeled political news dataset consisting of articles categorized into five political orientations.

### Classes

* Right
* Lean Right
* Center
* Lean Left
* Left

---

## Model Architecture

### Baseline

* TF-IDF Vectorizer
* Logistic Regression Classifier

### Final Model

* RoBERTa Base
* Hugging Face Transformers
* PyTorch

---

## Results

| Model                        |  Precision |     Recall | Weighted F1 |
| ---------------------------- | ---------: | ---------: | ----------: |
| TF-IDF + Logistic Regression |     76.63% |     73.98% |      73.48% |
| RoBERTa                      | **83.85%** | **82.11%** |  **82.12%** |

The transformer model significantly outperformed the traditional machine learning baseline, improving the weighted F1-score by nearly **9 percentage points**.

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Datasets
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Streamlit

---

## Installation

Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/Political-Bias-Analyzer.git

cd Political-Bias-Analyzer
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Train the transformer model

```bash
python src/train_transformer.py
```

The trained model will be saved inside

```
models/
```

---

## Evaluation

Evaluate the model

```bash
python src/evaluate_transformer.py
```

Outputs

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

---

## Inference

Run the command-line inference tool

```bash
python src/inference.py
```

Example

```
Enter article:

The government announced major investments in renewable energy and universal healthcare.

Prediction

Left

Confidence

91.42%
```

---

## Streamlit Web App

Launch the application

```bash
python -m streamlit run app/app.py
```

The application provides

* Political bias prediction
* Confidence score
* Probability distribution across all five classes

---

## Sample Predictions

### Example 1

Input

```
The government announced major investments in renewable energy, expanded universal healthcare, and stronger labor protections.
```

Prediction

```
Left
```

---

### Example 2

Input

```
The administration proposed reducing corporate taxes, increasing military spending, and strengthening border security.
```

Prediction

```
Right
```

---

## Future Improvements

* Explainable AI using SHAP
* Better handling of politically neutral articles
* Confidence calibration
* Interactive probability visualizations
* Hugging Face model hosting
* Streamlit Cloud deployment
* Multi-language support
* Larger and more balanced training dataset

---

## Limitations

* The model is trained on a relatively small labeled dataset.
* Performance on nuanced or highly mixed political articles may vary.
* The trained model is not included in this repository due to its size. Users should train the model locally or provide the saved model weights before running inference or the Streamlit application.

---

## Acknowledgements

* Hugging Face Transformers
* PyTorch
* Scikit-learn
* Streamlit

---

## License

This project is licensed under the MIT License.

---

## Author

**Akshat Sharma**

B.Tech, IIT Delhi

If you found this project useful, consider giving the repository a ⭐.
