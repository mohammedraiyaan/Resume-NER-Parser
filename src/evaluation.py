import os
import sys
import time
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

BENCHMARK_PATH = os.path.join(BASE_DIR, "artifacts", "benchmark_results.json")


def extract_chunks(tags: List[str]) -> List[Tuple[str, int, int]]:
    """
    Extract entity chunks (label, start_idx, end_idx) from BIO sequence.
    Strict entity-level chunk evaluation (exact CoNLL / seqeval methodology).
    """
    chunks = []
    chunk_type = None
    chunk_start = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if chunk_type is not None:
                chunks.append((chunk_type, chunk_start, i))
            chunk_type = tag[2:]
            chunk_start = i
        elif tag.startswith("I-"):
            type_i = tag[2:]
            if chunk_type == type_i:
                pass
            else:
                if chunk_type is not None:
                    chunks.append((chunk_type, chunk_start, i))
                chunk_type = type_i
                chunk_start = i
        else:
            if chunk_type is not None:
                chunks.append((chunk_type, chunk_start, i))
            chunk_type = None
            chunk_start = None

    if chunk_type is not None:
        chunks.append((chunk_type, chunk_start, len(tags)))

    return chunks

def evaluate_sequences(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, Any]:
    """
    Compute entity-level Precision, Recall, and F1 (seqeval compatible).
    Returns overall metrics and per-entity breakdown.
    """
    # Try seqeval first if available
    try:
        from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
        overall_p = precision_score(y_true, y_pred, zero_division=0)
        overall_r = recall_score(y_true, y_pred, zero_division=0)
        overall_f1 = f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        overall_p, overall_r, overall_f1 = None, None, None

    # Internal strict chunk calculation
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    for true_seq, pred_seq in zip(y_true, y_pred):
        true_chunks = set(extract_chunks(true_seq))
        pred_chunks = set(extract_chunks(pred_seq))

        for c in pred_chunks:
            if c in true_chunks:
                class_tp[c[0]] += 1
            else:
                class_fp[c[0]] += 1

        for c in true_chunks:
            if c not in pred_chunks:
                class_fn[c[0]] += 1

    all_classes = sorted(list(set(list(class_tp.keys()) + list(class_fp.keys()) + list(class_fn.keys()))))
    per_entity = {}
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())

    for c in all_classes:
        tp = class_tp[c]
        fp = class_fp[c]
        fn = class_fn[c]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        support = tp + fn
        per_entity[c] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": support
        }

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "overall": {
            "precision": round(overall_p if overall_p is not None else micro_p, 4),
            "recall": round(overall_r if overall_r is not None else micro_r, 4),
            "f1": round(overall_f1 if overall_f1 is not None else micro_f1, 4),
            "total_entities": total_tp + total_fn
        },
        "per_entity": per_entity
    }

def benchmark_models(models_dict: Dict[str, Any], test_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Benchmark multiple NER models on the identical test split.
    Records entity-level metrics, per-entity breakdown, and inference latency.
    """
    from src.tokenization_bio import tokenize_with_offsets, align_entities_to_bio

    # Prepare ground truth token sequences
    y_true_all = []
    token_sequences = []
    raw_texts = []

    for rec in test_records:
        content = rec["content"]
        raw_texts.append(content)
        entities = rec["entities"]
        bio_pairs = align_entities_to_bio(content, entities)
        token_sequences.append([p[0] for p in bio_pairs])
        y_true_all.append([p[1] for p in bio_pairs])

    results = {}

    for model_name, model_obj in models_dict.items():
        print(f"\nEvaluating {model_name} on {len(test_records)} test resumes...")
        y_pred_all = []
        start_time = time.time()

        for tokens, text in zip(token_sequences, raw_texts):
            if hasattr(model_obj, "predict_bio"):
                try:
                    preds = model_obj.predict_bio(tokens)
                except TypeError:
                    preds = model_obj.predict_bio(tokens, text)
            else:
                preds = ["O"] * len(tokens)
            y_pred_all.append(preds)

        elapsed = time.time() - start_time
        avg_latency_ms = (elapsed / len(test_records)) * 1000

        metrics = evaluate_sequences(y_true_all, y_pred_all)
        results[model_name] = {
            "precision": metrics["overall"]["precision"],
            "recall": metrics["overall"]["recall"],
            "f1": metrics["overall"]["f1"],
            "avg_latency_ms": round(avg_latency_ms, 2),
            "per_entity": metrics["per_entity"]
        }

        print(f"  {model_name} -> P: {metrics['overall']['precision']:.4f}, R: {metrics['overall']['recall']:.4f}, F1: {metrics['overall']['f1']:.4f}, Latency: {avg_latency_ms:.1f}ms/doc")

    os.makedirs(os.path.dirname(BENCHMARK_PATH), exist_ok=True)
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {BENCHMARK_PATH}")
    return results

if __name__ == "__main__":
    from src.models.crf_model import ResumeCRFModel
    from src.models.llm_model import ResumeLLMModel

    test_path = os.path.join(BASE_DIR, "data", "processed", "test.json")
    if not os.path.exists(test_path):
        from src.data_loader import split_and_save_data
        split_and_save_data()

    with open(test_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    crf = ResumeCRFModel()
    try:
        crf.load()
    except Exception:
        train_path = os.path.join(BASE_DIR, "data", "processed", "train.json")
        with open(train_path, "r", encoding="utf-8") as f:
            train_records = json.load(f)
        crf.train(train_records)

    llm = ResumeLLMModel()

    benchmark_models({
        "CRF (Hand-crafted Features)": crf,
        "LLM Few-shot Prompting": llm
    }, test_records)
