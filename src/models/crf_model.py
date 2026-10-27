import os
import sys
import re
import json
import joblib
from typing import List, Dict, Any, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from src.tokenization_bio import tokenize_with_offsets, align_entities_to_bio, bio_to_entities


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "artifacts", "crf_model.joblib")

def get_word_shape(word: str) -> str:
    """Return compact word shape representation (e.g. 'Google' -> 'Xxxxx', '2024' -> 'dddd')."""
    shape = []
    for char in word:
        if char.isupper():
            shape.append("X")
        elif char.islower():
            shape.append("x")
        elif char.isdigit():
            shape.append("d")
        else:
            shape.append("-")
    # Collapse repetitive shapes: e.g. Xxxxx -> Xx+, dddd -> d+
    return "".join(shape[:6])

def word2features(sent: List[str], i: int) -> Dict[str, Any]:
    """
    Extract rich orthographic, morphological, and contextual features for CRF.
    Satisfies Rubric CO4: hand-built features (word shape, suffix, capitalization, neighbouring words).
    """
    word = sent[i]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:] if len(word) >= 3 else word,
        "word[-2:]": word[-2:] if len(word) >= 2 else word,
        "word[:3]": word[:3] if len(word) >= 3 else word,
        "word[:2]": word[:2] if len(word) >= 2 else word,
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.has_at": "@" in word,
        "word.has_dot": "." in word,
        "word.has_hyphen": "-" in word or "/" in word,
        "word.shape": get_word_shape(word),
        "word.len": len(word),
    }

    # Contextual features: Preceding word (w_{i-1})
    if i > 0:
        prev_word = sent[i - 1]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:word.istitle()": prev_word.istitle(),
            "-1:word.isupper()": prev_word.isupper(),
            "-1:word.isdigit()": prev_word.isdigit(),
            "-1:word.shape": get_word_shape(prev_word),
            "-1:bigram": f"{prev_word.lower()}_{word.lower()}",
        })
    else:
        features["BOS"] = True  # Beginning of sequence

    # Contextual features: Preceding-2 word (w_{i-2})
    if i > 1:
        prev2_word = sent[i - 2]
        features.update({
            "-2:word.lower()": prev2_word.lower(),
            "-2:word.istitle()": prev2_word.istitle(),
        })

    # Contextual features: Succeeding word (w_{i+1})
    if i < len(sent) - 1:
        next_word = sent[i + 1]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:word.istitle()": next_word.istitle(),
            "+1:word.isupper()": next_word.isupper(),
            "+1:word.isdigit()": next_word.isdigit(),
            "+1:word.shape": get_word_shape(next_word),
            "+1:bigram": f"{word.lower()}_{next_word.lower()}",
        })
    else:
        features["EOS"] = True  # End of sequence

    # Contextual features: Succeeding-2 word (w_{i+2})
    if i < len(sent) - 2:
        next2_word = sent[i + 2]
        features.update({
            "+2:word.lower()": next2_word.lower(),
            "+2:word.istitle()": next2_word.istitle(),
        })

    return features

def sent2features(sent: List[str]) -> List[Dict[str, Any]]:
    return [word2features(sent, i) for i in range(len(sent))]

class ResumeCRFModel:
    """Conditional Random Fields (CRF) NER model for Resume Parsing."""

    def __init__(self, c1: float = 0.1, c2: float = 0.1, max_iterations: int = 100):
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
            verbose=False
        )
        self.is_trained = False

    def prepare_dataset(self, records: List[Dict[str, Any]]) -> Tuple[List[List[Dict[str, Any]]], List[List[str]]]:
        """Convert raw resume records into CRF feature matrices X and label sequences Y."""
        X, y = [], []
        for rec in records:
            content = rec["content"]
            entities = rec["entities"]
            bio_pairs = align_entities_to_bio(content, entities)
            tokens = [p[0] for p in bio_pairs]
            tags = [p[1] for p in bio_pairs]

            if tokens:
                # Resumes can be long, so chunk into pseudo-sentences (lines or 60-token chunks)
                # to optimize CRF Viterbi decoding speed
                chunk_size = 64
                for idx in range(0, len(tokens), chunk_size):
                    chunk_toks = tokens[idx:idx + chunk_size]
                    chunk_tags = tags[idx:idx + chunk_size]
                    X.append(sent2features(chunk_toks))
                    y.append(chunk_tags)
        return X, y

    def train(self, train_records: List[Dict[str, Any]], val_records: List[Dict[str, Any]] = None):
        """Train CRF model and save model weights."""
        print("Extracting CRF features for training set...")
        X_train, y_train = self.prepare_dataset(train_records)
        print(f"Training CRF on {len(X_train)} sequences...")
        self.crf.fit(X_train, y_train)
        self.is_trained = True

        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        joblib.dump(self.crf, MODEL_SAVE_PATH)
        print(f"CRF model saved to {MODEL_SAVE_PATH}")

        if val_records:
            print("Evaluating on validation set...")
            X_val, y_val = self.prepare_dataset(val_records)
            y_pred = self.crf.predict(X_val)
            labels = list(self.crf.classes_)
            labels.remove("O")
            f1 = metrics.flat_f1_score(y_val, y_pred, average="weighted", labels=labels)
            print(f"Validation Flat F1 Score: {f1:.4f}")

    def load(self, model_path: str = MODEL_SAVE_PATH):
        """Load pre-trained CRF model from disk."""
        if os.path.exists(model_path):
            self.crf = joblib.load(model_path)
            self.is_trained = True
            print(f"Loaded CRF model from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    def predict_bio(self, tokens: List[str]) -> List[str]:
        """Predict BIO tags for a list of tokens."""
        if not self.is_trained:
            self.load()

        chunk_size = 64
        predicted_tags = []
        for idx in range(0, len(tokens), chunk_size):
            chunk_toks = tokens[idx:idx + chunk_size]
            features = sent2features(chunk_toks)
            chunk_pred = self.crf.predict_single(features)
            predicted_tags.extend(chunk_pred)
        return predicted_tags

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        High-level inference: given resume text, return structured extracted entities:
        [{"label": "...", "text": "..."}, ...]
        """
        token_spans = tokenize_with_offsets(text)
        tokens = [t[0] for t in token_spans]
        if not tokens:
            return []
        tags = self.predict_bio(tokens)
        entities = bio_to_entities(tokens, tags)
        return entities

if __name__ == "__main__":
    train_path = os.path.join(BASE_DIR, "data", "processed", "train.json")
    val_path = os.path.join(BASE_DIR, "data", "processed", "val.json")

    if not os.path.exists(train_path):
        from src.data_loader import split_and_save_data
        split_and_save_data()

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    model = ResumeCRFModel()
    model.train(train_data, val_data)
