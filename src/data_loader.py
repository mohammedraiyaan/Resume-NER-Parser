import os
import json
import shutil
import urllib.request
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split

DATA_URL = "https://raw.githubusercontent.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy/master/traindata.json"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Entity Recognition in Resumes.json")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TEST_RESUMES_DIR = os.path.join(BASE_DIR, "data", "test_resumes")

# Standard entity classes in the Kaggle / DataTurks dataset
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

def ensure_directories():
    os.makedirs(os.path.join(BASE_DIR, "data", "raw"), exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(TEST_RESUMES_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "artifacts"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "docs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "notebooks"), exist_ok=True)

def download_raw_data() -> str:
    """Download the raw DataTurks / Kaggle dataset if not already present."""
    ensure_directories()
    if not os.path.exists(RAW_DATA_PATH) or os.path.getsize(RAW_DATA_PATH) == 0:
        print(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
        print(f"Saved raw data to {RAW_DATA_PATH}")
    else:
        print(f"Raw dataset already exists at {RAW_DATA_PATH}")
    return RAW_DATA_PATH

def clean_annotations(content: str, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean and sanitize raw annotations:
    - Normalizes label names.
    - Drops 'UNKNOWN' or invalid labels.
    - Trims leading/trailing whitespace from entity text and shifts start/end offsets accordingly.
    - Resolves overlapping character spans by preserving the earlier or longer span.
    """
    cleaned = []
    if not annotations:
        return cleaned

    for ann in annotations:
        if not ann or "points" not in ann:
            continue
        raw_labels = ann.get("label", [])
        if isinstance(raw_labels, list) and len(raw_labels) > 0:
            label = raw_labels[0].strip()
        elif isinstance(raw_labels, str):
            label = raw_labels.strip()
        else:
            continue

        # Filter out noisy non-canonical labels
        if label not in CANONICAL_ENTITIES:
            continue

        for pt in ann.get("points", []):
            start = pt.get("start")
            end = pt.get("end")
            text = pt.get("text", "")

            if start is None or end is None:
                continue

            # DataTurks points end is inclusive. Let's adjust to python slice end = end + 1 if needed
            # In DataTurks: content[start:end+1] or content[start:end]
            # Let's verify against content
            target_slice = content[start:end]
            target_slice_plus1 = content[start:end+1]

            if target_slice == text:
                real_end = end
            elif target_slice_plus1 == text:
                real_end = end + 1
            else:
                # Search nearby (within +/- 3 chars)
                found = False
                for delta in range(-3, 4):
                    s = max(0, start + delta)
                    e = s + len(text)
                    if content[s:e] == text:
                        start, real_end = s, e
                        found = True
                        break
                if not found:
                    real_end = end + 1

            extracted_text = content[start:real_end]
            
            # Trim leading whitespace
            lead_ws = len(extracted_text) - len(extracted_text.lstrip())
            start += lead_ws
            extracted_text = extracted_text.lstrip()

            # Trim trailing whitespace
            trail_ws = len(extracted_text) - len(extracted_text.rstrip())
            real_end -= trail_ws
            extracted_text = extracted_text.rstrip()

            if start < real_end and len(extracted_text) > 0:
                cleaned.append({
                    "label": label,
                    "start": start,
                    "end": real_end,
                    "text": extracted_text
                })

    # Sort spans by start asc, end desc
    cleaned.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    # Resolve overlapping spans (keep longest/first)
    non_overlapping = []
    last_end = -1
    for span in cleaned:
        if span["start"] >= last_end:
            non_overlapping.append(span)
            last_end = span["end"]

    return non_overlapping

def load_and_clean_data(file_path: str = None) -> List[Dict[str, Any]]:
    """Load JSON lines, parse content and annotations, and clean offsets."""
    if file_path is None:
        file_path = download_raw_data()

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = data.get("content", "")
            if not content.strip():
                continue

            raw_annotations = data.get("annotation", [])
            cleaned_ann = clean_annotations(content, raw_annotations)

            records.append({
                "id": idx,
                "content": content,
                "entities": cleaned_ann
            })

    print(f"Successfully loaded and cleaned {len(records)} resume records.")
    return records

def split_and_save_data(test_size: float = 0.10, val_size: float = 0.10, random_state: int = 42) -> Tuple[str, str, str]:
    """
    Split records into 80% train, 10% validation, 10% test with fixed seed.
    Save to data/processed/{train,val,test}.json
    """
    ensure_directories()
    records = load_and_clean_data()
    total = len(records)

    # First split: train + val vs test (10%)
    train_val, test = train_test_split(records, test_size=test_size, random_state=random_state)
    # Second split: train vs val (10% of total -> val_size / (1 - test_size))
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val_size, random_state=random_state)

    train_path = os.path.join(PROCESSED_DIR, "train.json")
    val_path = os.path.join(PROCESSED_DIR, "val.json")
    test_path = os.path.join(PROCESSED_DIR, "test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

    print(f"Data split saved successfully:")
    print(f"  Train : {len(train)} resumes ({len(train)/total*100:.1f}%) -> {train_path}")
    print(f"  Val   : {len(val)} resumes ({len(val)/total*100:.1f}%) -> {val_path}")
    print(f"  Test  : {len(test)} resumes ({len(test)/total*100:.1f}%) -> {test_path}")

    # Copy 1BM23AI103_Resume.pdf to test_resumes if available
    download_bmsce_resume = r"C:\Users\Administrator\Downloads\1BM23AI103_Resume.pdf"
    if os.path.exists(download_bmsce_resume):
        target_bmsce = os.path.join(TEST_RESUMES_DIR, "1BM23AI103_Resume.pdf")
        shutil.copyfile(download_bmsce_resume, target_bmsce)
        print(f"Copied test resume to {target_bmsce}")

    return train_path, val_path, test_path

if __name__ == "__main__":
    split_and_save_data()
