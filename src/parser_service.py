import os
import io
import time
import json
import re
from typing import List, Dict, Any, Union, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def extract_text_from_pdf(pdf_source: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extract clean, structured text from PDF using pdfplumber (with pypdf fallback).
    Preserves document layout and removes extraction artifacts.
    """
    extracted_pages = []

    # Try pdfplumber first
    try:
        import pdfplumber
        if isinstance(pdf_source, (bytes, bytearray)):
            pdf_source = io.BytesIO(pdf_source)

        with pdfplumber.open(pdf_source) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=True) or page.extract_text()
                if text:
                    extracted_pages.append(text)
        if extracted_pages:
            raw_text = "\n\n".join(extracted_pages)
            return clean_extracted_text(raw_text)
    except Exception:
        pass

    # Fallback to pypdf
    try:
        import pypdf
        if isinstance(pdf_source, (bytes, bytearray)):
            pdf_source = io.BytesIO(pdf_source)
        reader = pypdf.PdfReader(pdf_source)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_pages.append(text)
        if extracted_pages:
            raw_text = "\n\n".join(extracted_pages)
            return clean_extracted_text(raw_text)
    except Exception as e:
        print(f"PDF extraction error: {e}")

    return ""

def clean_extracted_text(text: str) -> str:
    """Normalize whitespace, remove non-printable characters and PDF artifacts."""
    if not text:
        return ""
    # Normalize common ligatures
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("’", "'").replace("“", '"').replace("”", '"')
    # Remove null bytes and non-printable control chars
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Collapse 3+ consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

class ResumeParserService:
    """Unified multi-model Resume Parsing Service."""

    def __init__(self):
        self._crf = None
        self._spacy = None
        self._bert = None
        self._llm = None

    def get_crf_model(self):
        if self._crf is None:
            from src.models.crf_model import ResumeCRFModel
            self._crf = ResumeCRFModel()
            try:
                self._crf.load()
            except Exception as e:
                print(f"Could not load pre-trained CRF: {e}. Model will train on first use.")
        return self._crf

    def get_spacy_model(self):
        if self._spacy is None:
            from src.models.spacy_model import ResumeSpacyModel
            self._spacy = ResumeSpacyModel()
            try:
                self._spacy.load()
            except Exception as e:
                print(f"spaCy model not yet trained or loaded: {e}")
        return self._spacy

    def get_bert_model(self):
        if self._bert is None:
            from src.models.bert_model import ResumeBERTModel
            self._bert = ResumeBERTModel()
            self._bert.load_tokenizer_and_model()
        return self._bert

    def get_llm_model(self):
        if self._llm is None:
            from src.models.llm_model import ResumeLLMModel
            self._llm = ResumeLLMModel()
        return self._llm

    def parse_text(self, text: str, model_type: str = "crf") -> Dict[str, Any]:
        """
        Parse text with chosen model ('crf', 'spacy', 'bert', 'llm', or 'all').
        Returns standardized structured JSON output.
        """
        cleaned_text = clean_extracted_text(text)
        if not cleaned_text:
            return {"error": "Empty text provided.", "entities": []}

        start_time = time.time()
        model_type = model_type.lower()

        if model_type == "all":
            # Run all 4 models and produce comparative output
            results = {}
            for m in ["crf", "spacy", "bert", "llm"]:
                try:
                    res = self.parse_text(cleaned_text, model_type=m)
                    results[m] = res
                except Exception as e:
                    results[m] = {"error": str(e), "entities": []}
            return {
                "text_snippet": cleaned_text[:300] + "...",
                "models": results,
                "total_time_ms": round((time.time() - start_time) * 1000, 2)
            }

        # Single model execution
        entities = []
        if model_type == "crf":
            model = self.get_crf_model()
            entities = model.predict(cleaned_text)
        elif model_type == "spacy":
            model = self.get_spacy_model()
            entities = model.predict(cleaned_text)
        elif model_type == "bert":
            model = self.get_bert_model()
            entities = model.predict(cleaned_text)
        elif model_type == "llm":
            model = self.get_llm_model()
            entities = model.predict(cleaned_text)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # Group by entity class
        grouped = {ent: [] for ent in CANONICAL_ENTITIES}
        for item in entities:
            label = item.get("label")
            txt = item.get("text", "").strip()
            if label in grouped and txt and txt not in grouped[label]:
                grouped[label].append(txt)

        return {
            "model_used": model_type.upper(),
            "latency_ms": elapsed_ms,
            "entity_count": len(entities),
            "grouped_entities": grouped,
            "raw_entities": entities,
            "text_length_chars": len(cleaned_text)
        }

    def parse_pdf(self, pdf_source: Union[str, bytes, io.BytesIO], model_type: str = "crf") -> Dict[str, Any]:
        """Extract text from PDF and parse entities."""
        text = extract_text_from_pdf(pdf_source)
        if not text:
            return {"error": "Could not extract readable text from PDF.", "entities": []}
        parsed = self.parse_text(text, model_type=model_type)
        parsed["source_type"] = "PDF"
        return parsed
