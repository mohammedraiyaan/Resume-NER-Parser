import os
import sys
import json
import random
from typing import List, Dict, Any, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.tokenization_bio import tokenize_with_offsets, bio_to_entities

try:
    import spacy
    from spacy.training import Example
    from spacy.util import minibatch, compounding
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

SPACY_SAVE_DIR = os.path.join(BASE_DIR, "artifacts", "spacy_ner")


CANONICAL_ENTITIES = [
    "Name",
    "College Name",
    "Degree",
    "Graduation Year",
    "Years of Experience",
    "Companies worked at",
    "Designation",
    "Skills",
    "Location",
    "Email Address"
]

class ResumeSpacyModel:
    """Baseline statistical transition-based NER pipeline using spaCy."""

    def __init__(self, model_dir: str = SPACY_SAVE_DIR):
        self.model_dir = model_dir
        self.nlp = None

    def initialize_new_model(self):
        """Create a blank English pipeline and add the NER pipe."""
        self.nlp = spacy.blank("en")
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")

        for label in CANONICAL_ENTITIES:
            ner.add_label(label)
        return self.nlp

    def _prepare_examples(self, records: List[Dict[str, Any]]) -> List[Example]:
        """Convert dataset records into spaCy Training Examples with valid character spans."""
        examples = []
        for rec in records:
            content = rec["content"]
            entities = rec["entities"]
            spans = []
            for ent in entities:
                start = ent["start"]
                end = ent["end"]
                label = ent["label"]
                if label in CANONICAL_ENTITIES:
                    spans.append((start, end, label))

            doc = self.nlp.make_doc(content)
            # Filter conflicting or misaligned spans
            example = Example.from_dict(doc, {"entities": spans})
            examples.append(example)
        return examples

    def train(self, train_records: List[Dict[str, Any]], n_iter: int = 25):
        """Train the spaCy NER pipeline on the training split."""
        print("Initializing spaCy pipeline...")
        self.initialize_new_model()

        examples = self._prepare_examples(train_records)
        print(f"Training spaCy statistical NER on {len(examples)} resumes for {n_iter} iterations...")

        # Initialize pipeline weights
        optimizer = self.nlp.initialize(lambda: examples)

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            sizes = compounding(4.0, 32.0, 1.001)
            for itn in range(n_iter):
                random.shuffle(examples)
                batches = minibatch(examples, size=sizes)
                losses = {}
                for batch in batches:
                    self.nlp.update(batch, drop=0.2, sgd=optimizer, losses=losses)
                if (itn + 1) % 5 == 0 or itn == n_iter - 1:
                    print(f"  Iteration {itn + 1:2d}/{n_iter} - Loss: {losses.get('ner', 0.0):.2f}")

        os.makedirs(self.model_dir, exist_ok=True)
        self.nlp.to_disk(self.model_dir)
        print(f"Saved trained spaCy NER model to {self.model_dir}")

    def load(self, model_dir: str = None):
        """Load trained spaCy model from disk."""
        target_dir = model_dir or self.model_dir
        if os.path.exists(target_dir):
            self.nlp = spacy.load(target_dir)
            print(f"Loaded spaCy model from {target_dir}")
        else:
            raise FileNotFoundError(f"spaCy model directory not found at {target_dir}")

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from raw resume text using the trained spaCy model."""
        if self.nlp is None:
            self.load()
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in CANONICAL_ENTITIES:
                entities.append({
                    "label": ent.label_,
                    "text": ent.text.strip(),
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        return entities

    def predict_bio(self, tokens: List[str], text: str = None) -> List[str]:
        """Align spaCy entity predictions to input tokens for seqeval evaluation."""
        if self.nlp is None:
            self.load()
        if text is None:
            text = " ".join(tokens)

        doc = self.nlp(text)
        token_spans = tokenize_with_offsets(text)

        tags = ["O"] * len(token_spans)
        for ent in doc.ents:
            if ent.label_ not in CANONICAL_ENTITIES:
                continue
            e_start, e_end = ent.start_char, ent.end_char
            first = True
            for i, (tok, t_s, t_e) in enumerate(token_spans):
                if t_s >= e_start and t_e <= e_end:
                    if first:
                        tags[i] = f"B-{ent.label_}"
                        first = False
                    else:
                        tags[i] = f"I-{ent.label_}"
        return tags

if __name__ == "__main__":
    train_path = os.path.join(BASE_DIR, "data", "processed", "train.json")
    if not os.path.exists(train_path):
        from src.data_loader import split_and_save_data
        split_and_save_data()

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    model = ResumeSpacyModel()
    model.train(train_data, n_iter=20)
