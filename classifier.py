"""Model loader and inference engine for Consumer Complaint Classification."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "saved_model"))

DEPARTMENTS: Dict[str, Dict[str, str]] = {
    "credit_reporting": {
        "title": "Credit Reporting & Bureau Disputes",
        "queue": "CR-01",
        "sla": "24 Hours",
        "desc": "Credit file inaccuracies, unauthorized inquiries, and bureau dispute escalations.",
    },
    "debt_collection": {
        "title": "Debt Collection & Compliance",
        "queue": "DC-02",
        "sla": "12 Hours",
        "desc": "Third-party collection conduct, disputed balances, and harassment reports.",
    },
    "credit_card": {
        "title": "Credit Card Services & Billing",
        "queue": "CC-03",
        "sla": "24 Hours",
        "desc": "Unauthorized card charges, APR/fee disputes, and statement billing issues.",
    },
    "mortgages_and_loans": {
        "title": "Mortgages & Consumer Lending",
        "queue": "ML-04",
        "sla": "48 Hours",
        "desc": "Escrow adjustments, loan servicing errors, and repayment schedule inquiries.",
    },
    "retail_banking": {
        "title": "Retail Banking & Fraud Ops",
        "queue": "RB-05",
        "sla": "6 Hours",
        "desc": "Checking/savings accounts, ATM/debit fraud, and wire transfer disputes.",
    },
}


def clean_text_basic(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class ComplaintClassifier:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        meta_path = model_dir / "model_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}. Place saved_model/ in project root.")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.model_type = self.meta["model_type"]
        self.model_name = self.meta.get("model_name", self.model_type)
        self.id2label = {int(k): v for k, v in self.meta["id2label"].items()}

        if self.model_type == "transformer":
            self._load_transformer()
        elif self.model_type == "keras":
            self._load_keras()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _load_transformer(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tf_dir = self.model_dir / "transformer"
        self.tokenizer = AutoTokenizer.from_pretrained(tf_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(tf_dir)
        self.model.to(self.device)
        self.model.eval()
        self.max_len = int(self.meta.get("max_len", 256))

    def _load_keras(self) -> None:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        self._pad_sequences = pad_sequences
        self.model = tf.keras.models.load_model(self.model_dir / "keras_model.keras")
        with open(self.model_dir / "tokenizer.json", "r", encoding="utf-8") as f:
            self.tokenizer = tokenizer_from_json(f.read())
        self.max_len = int(self.meta.get("max_len", 200))

    def predict(self, complaint_text: str) -> Dict[str, Any]:
        if self.model_type == "transformer":
            import torch

            inputs = self.tokenizer(
                complaint_text,
                truncation=True,
                padding=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        else:
            cleaned = clean_text_basic(complaint_text)
            seq = self.tokenizer.texts_to_sequences([cleaned])
            padded = self._pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")
            probs = self.model.predict(padded, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        label = self.id2label[pred_idx]
        confidence = float(probs[pred_idx])
        dept = DEPARTMENTS.get(label, {"title": label, "queue": "GEN", "sla": "24 Hours", "desc": ""})

        breakdown = sorted(
            [
                {
                    "label": self.id2label[i],
                    "title": DEPARTMENTS.get(self.id2label[i], {}).get("title", self.id2label[i]),
                    "score": round(float(probs[i]) * 100, 2),
                }
                for i in range(len(probs))
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        return {
            "label": label,
            "department": dept["title"],
            "queue": dept["queue"],
            "sla": dept["sla"],
            "description": dept["desc"],
            "confidence": round(confidence * 100, 2),
            "auto_route": confidence >= 0.70,
            "breakdown": breakdown,
            "model_name": self.model_name,
        }


def get_benchmark_rows(model_dir: Path = MODEL_DIR) -> List[Dict[str, Any]]:
    csv_path = model_dir / "model_comparison.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path).to_dict(orient="records")
    return [
        {"Model": "DistilBERT", "Accuracy": 0.8537, "Precision (Weighted)": 0.8573, "Recall (Weighted)": 0.8537, "F1-Score (Weighted)": 0.8547, "F1-Score (Macro)": 0.8425},
        {"Model": "GRU", "Accuracy": 0.8357, "Precision (Weighted)": 0.8406, "Recall (Weighted)": 0.8357, "F1-Score (Weighted)": 0.8366, "F1-Score (Macro)": 0.8239},
        {"Model": "LSTM", "Accuracy": 0.8350, "Precision (Weighted)": 0.8411, "Recall (Weighted)": 0.8350, "F1-Score (Weighted)": 0.8360, "F1-Score (Macro)": 0.8232},
        {"Model": "SimpleRNN", "Accuracy": 0.6907, "Precision (Weighted)": 0.7125, "Recall (Weighted)": 0.6907, "F1-Score (Weighted)": 0.6958, "F1-Score (Macro)": 0.6403},
    ]
