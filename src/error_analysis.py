import os
import sys
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

ERROR_REPORT_PATH = os.path.join(BASE_DIR, "artifacts", "error_analysis_report.json")


from src.evaluation import extract_chunks
from src.tokenization_bio import align_entities_to_bio

def analyze_model_errors(model_name: str, y_true: List[List[str]], y_pred: List[List[str]], tokens_list: List[List[str]]) -> Dict[str, Any]:
    """
    Categorize model prediction errors into formal error taxonomy:
    - Boundary Mismatches (exact vs partial match)
    - Label Confusion (e.g. Skills vs Companies)
    - False Positives (over-tagging)
    - False Negatives (missed entities)
    - Average span length analysis
    """
    total_true = 0
    total_pred = 0
    exact_matches = 0
    boundary_errors = 0
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    missed_by_class = Counter()
    spurious_by_class = Counter()
    span_lengths_by_class = defaultdict(list)

    for true_seq, pred_seq, tokens in zip(y_true, y_pred, tokens_list):
        true_chunks = extract_chunks(true_seq)
        pred_chunks = extract_chunks(pred_seq)

        total_true += len(true_chunks)
        total_pred += len(pred_chunks)

        for c_type, c_s, c_e in true_chunks:
            span_lengths_by_class[c_type].append(c_e - c_s)

        matched_true = set()
        matched_pred = set()

        # Check for exact matches
        for i, (t_type, t_s, t_e) in enumerate(true_chunks):
            for j, (p_type, p_s, p_e) in enumerate(pred_chunks):
                if j in matched_pred or i in matched_true:
                    continue
                if t_type == p_type and t_s == p_s and t_e == p_e:
                    exact_matches += 1
                    matched_true.add(i)
                    matched_pred.add(j)

        # Check for boundary errors (same type, overlapping span)
        for i, (t_type, t_s, t_e) in enumerate(true_chunks):
            if i in matched_true:
                continue
            for j, (p_type, p_s, p_e) in enumerate(pred_chunks):
                if j in matched_pred:
                    continue
                # Overlap condition: max(t_s, p_s) < min(t_e, p_e)
                if t_type == p_type and max(t_s, p_s) < min(t_e, p_e):
                    boundary_errors += 1
                    matched_true.add(i)
                    matched_pred.add(j)
                    break

        # Check for label confusion (different type, overlapping span)
        for i, (t_type, t_s, t_e) in enumerate(true_chunks):
            if i in matched_true:
                continue
            for j, (p_type, p_s, p_e) in enumerate(pred_chunks):
                if j in matched_pred:
                    continue
                if max(t_s, p_s) < min(t_e, p_e):
                    confusion_matrix[t_type][p_type] += 1
                    matched_true.add(i)
                    matched_pred.add(j)
                    break

        # False negatives (unmatched true chunks)
        for i, (t_type, t_s, t_e) in enumerate(true_chunks):
            if i not in matched_true:
                missed_by_class[t_type] += 1

        # False positives (unmatched pred chunks)
        for j, (p_type, p_s, p_e) in enumerate(pred_chunks):
            if j not in matched_pred:
                spurious_by_class[p_type] += 1

    avg_span_lengths = {c: round(sum(l) / len(l), 2) for c, l in span_lengths_by_class.items() if l}

    # Convert nested dict
    conf_serializable = {k: dict(v) for k, v in confusion_matrix.items()}

    return {
        "model_name": model_name,
        "total_ground_truth_entities": total_true,
        "total_predicted_entities": total_pred,
        "exact_matches": exact_matches,
        "boundary_mismatches": boundary_errors,
        "exact_match_ratio": round(exact_matches / total_true, 4) if total_true > 0 else 0.0,
        "missed_entities_fn": dict(missed_by_class),
        "hallucinated_entities_fp": dict(spurious_by_class),
        "label_confusion": conf_serializable,
        "average_entity_token_length": avg_span_lengths
    }

def run_error_analysis(test_records: List[Dict[str, Any]], crf_model=None, spacy_model=None) -> Dict[str, Any]:
    """Execute complete error analysis across models and output formal report."""
    y_true_all = []
    tokens_list = []

    for rec in test_records:
        content = rec["content"]
        entities = rec["entities"]
        bio_pairs = align_entities_to_bio(content, entities)
        tokens_list.append([p[0] for p in bio_pairs])
        y_true_all.append([p[1] for p in bio_pairs])

    reports = {}

    if crf_model:
        y_pred_crf = [crf_model.predict_bio(toks) for toks in tokens_list]
        reports["CRF"] = analyze_model_errors("CRF", y_true_all, y_pred_crf, tokens_list)

    os.makedirs(os.path.dirname(ERROR_REPORT_PATH), exist_ok=True)
    with open(ERROR_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    print(f"Error analysis written to {ERROR_REPORT_PATH}")
    return reports

if __name__ == "__main__":
    from src.models.crf_model import ResumeCRFModel

    test_path = os.path.join(BASE_DIR, "data", "processed", "test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    crf = ResumeCRFModel()
    crf.load()
    run_error_analysis(test_records, crf_model=crf)
