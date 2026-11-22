# Consumer Complaint Classification (NLP Deep Learning)

End-to-end NLP text classification pipeline that compares **SimpleRNN**, **LSTM**, **GRU**, and a **Fine-tuned HuggingFace Transformer (`distilbert-base-uncased`)** on the [Kaggle Consumer Complaints Dataset](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp), and deploys the best-performing model with **Gradio**.

## Project Structure

```
consumer-complaint-classification/
├── consumer_complaint_classification.ipynb   # Full end-to-end training, evaluation & comparison notebook
├── app.py                                    # Standalone Gradio web app for deployment
├── requirements.txt                          # Python dependencies
└── README.md
```

## Quick Start (Google Colab / Kaggle — Recommended)

1. Upload `consumer_complaint_classification.ipynb` to **Google Colab** (enable GPU: `Runtime -> Change runtime type -> T4 GPU`) or **Kaggle Notebooks** (enable `GPU T4 x2` / `P100`).
2. Run all cells in order:
   - Downloads the Kaggle dataset automatically via `kagglehub` (or loads it directly from `/kaggle/input/...` on Kaggle).
   - Explores the dataset, visualizes class imbalance & narrative lengths.
   - Preprocesses text (lowercasing, removing punctuation/numbers/special chars, stopword removal, optional lemmatization).
   - Handles class imbalance with `class_weight="balanced"` (Keras) and weighted `CrossEntropyLoss` (Transformer).
   - Trains & evaluates **SimpleRNN**, **LSTM**, **GRU**, and **DistilBERT**.
   - Compares all 4 models (Accuracy, Precision, Recall, F1-score, Confusion Matrices).
   - Saves the best model to `saved_model/` and launches an interactive Gradio demo.

## Running the Standalone Gradio App Locally

After exporting the `saved_model/` folder from Colab/Kaggle into this directory:

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860` in your browser.
