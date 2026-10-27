import re
from typing import List, Dict, Any, Tuple

def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize text using regex, returning (token, start_char, end_char).
    Splits words and punctuation while tracking character offsets.
    """
    tokens = []
    # Match words or individual non-whitespace characters (punctuation)
    for match in re.finditer(r'\S+', text):
        raw_word = match.group(0)
        start = match.start()
        end = match.end()

        # Further split punctuation at boundaries if needed
        # Simple sub-regex to separate leading/trailing punctuation cleanly
        sub_tokens = list(re.finditer(r'\w+|[^\w\s]', raw_word))
        if sub_tokens:
            for st in sub_tokens:
                sub_str = st.group(0)
                sub_start = start + st.start()
                sub_end = start + st.end()
                tokens.append((sub_str, sub_start, sub_end))
        else:
            tokens.append((raw_word, start, end))
    return tokens

def align_entities_to_bio(text: str, entities: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Map character-level entity spans to token-level BIO tags.
    Returns a list of (token, bio_tag) pairs.
    """
    token_spans = tokenize_with_offsets(text)
    labeled_tokens = []

    # Map each char position to its entity if present
    # To be efficient, sort entities by start
    entities_sorted = sorted(entities, key=lambda x: x["start"])

    for token, t_start, t_end in token_spans:
        assigned_tag = "O"

        for ent in entities_sorted:
            e_start = ent["start"]
            e_end = ent["end"]
            label = ent["label"]

            # Check overlap between [t_start, t_end] and [e_start, e_end]
            if t_start >= e_start and t_end <= e_end:
                if t_start == e_start or not assigned_tag.startswith("B-"):
                    # First token in entity
                    # If this is the beginning of the entity span, B-
                    # Check if previous tokens in this entity have already been tagged
                    if t_start == e_start:
                        assigned_tag = f"B-{label}"
                    else:
                        # Could be first token if leading whitespace was stripped
                        assigned_tag = f"I-{label}"
                else:
                    assigned_tag = f"I-{label}"
                break
            elif t_start < e_end and t_end > e_start:
                # Partial overlap fallback
                assigned_tag = f"B-{label}" if t_start <= e_start else f"I-{label}"
                break

        labeled_tokens.append((token, assigned_tag))

    # Fix any isolated I- tags (ensure every I- is preceded by B- or I- of same label)
    corrected = []
    for i, (tok, tag) in enumerate(labeled_tokens):
        if tag.startswith("I-"):
            label_type = tag[2:]
            prev_tag = corrected[-1][1] if corrected else "O"
            if not (prev_tag == f"B-{label_type}" or prev_tag == f"I-{label_type}"):
                tag = f"B-{label_type}"
        corrected.append((tok, tag))

    return corrected

def bio_to_entities(tokens: List[str], tags: List[str]) -> List[Dict[str, Any]]:
    """
    Convert a sequence of tokens and BIO tags back into structured entity dictionaries:
    [{"label": "Skills", "text": "Python, Java"}, ...]
    """
    entities = []
    current_label = None
    current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_label and current_tokens:
                entities.append({
                    "label": current_label,
                    "text": " ".join(current_tokens)
                })
            current_label = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_label and tag[2:] == current_label:
            current_tokens.append(token)
        else:
            if current_label and current_tokens:
                entities.append({
                    "label": current_label,
                    "text": " ".join(current_tokens)
                })
            current_label = None
            current_tokens = []

    if current_label and current_tokens:
        entities.append({
            "label": current_label,
            "text": " ".join(current_tokens)
        })

    return entities

def save_conll_file(records: List[Dict[str, Any]], output_path: str):
    """Save records in CoNLL format: one token and tag per line, blank line between documents."""
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            content = rec["content"]
            entities = rec["entities"]
            bio_pairs = align_entities_to_bio(content, entities)
            for token, tag in bio_pairs:
                f.write(f"{token} {tag}\n")
            f.write("\n")  # Document boundary
    print(f"Saved CoNLL dataset to {output_path}")

if __name__ == "__main__":
    import os
    import json
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, "data", "processed")

    for split in ["train", "val", "test"]:
        json_path = os.path.join(proc_dir, f"{split}.json")
        conll_path = os.path.join(proc_dir, f"conll_{split}.txt")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            save_conll_file(data, conll_path)

