import json
import logging
from typing import List, Dict, Any
from misc.metrics import compute_bleu, compute_rouge

logger = logging.getLogger(__name__)


def compute_metrics_for_annotations(auto_results: List[Dict[str, Any]], raw_data: List[Dict[str, Any]]):
    """
    Compute and log BLEU and ROUGE scores for automatically annotated results.
    """
    id2raw = {str(d.get('id', i)): d for i, d in enumerate(raw_data)}
    bleu_scores = []
    rouge_1 = []
    rouge_2 = []
    rouge_l = []
    total = 0
    for r in auto_results:
        rid = str(r.get('id', None))
        raw = id2raw.get(rid, {})
        gt = raw.get('answer', "")
        if isinstance(gt, list):
            gt = gt[0] if gt else ""
        pred = r.get('annotation', "").strip()
        ref = str(gt).strip()
        if pred and ref:
            bleu = compute_bleu(ref, pred)
            rouge = compute_rouge(ref, pred)
            bleu_scores.append(bleu)
            rouge_1.append(rouge['rouge-1']['f'])
            rouge_2.append(rouge['rouge-2']['f'])
            rouge_l.append(rouge['rouge-l']['f'])
            total += 1
    avg_bleu = sum(bleu_scores) / total if total > 0 else 0.0
    avg_rouge_1 = sum(rouge_1) / total if total > 0 else 0.0
    avg_rouge_2 = sum(rouge_2) / total if total > 0 else 0.0
    avg_rouge_l = sum(rouge_l) / total if total > 0 else 0.0
    logger.info("Auto-annotation BLEU-4: %.4f", avg_bleu)
    logger.info("Auto-annotation ROUGE-1: %.4f  ROUGE-2: %.4f  ROUGE-L: %.4f  (Total %d)", avg_rouge_1, avg_rouge_2, avg_rouge_l, total)


def export_annotation_results(results: List[Dict[str, Any]], raw_data: List[Dict[str, Any]], output_path: str = "final_annotation_results.json"):
    """
    Merge annotation results with raw data and export to a JSON file.
    :param results: List of annotation results, each item is a dict and must contain an 'id' field
    :param raw_data: List of raw data, each item is a dict and must contain 'id', 'question', 'context', 'answer', etc.
    :param output_path: Path to export the JSON file
    """
    id2raw = {str(d.get('id', i)): d for i, d in enumerate(raw_data)}
    export_data = []
    for r in results:
        rid = str(r.get('id', None))
        raw = id2raw.get(rid, {})
        export_data.append({
            "id": rid,
            "question": raw.get('question', ""),
            "context": raw.get('context', ""),
            "route": r.get('route', ""),
            "annotation": r.get('annotation', "")
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    logger.info("Exported to %s", output_path)
