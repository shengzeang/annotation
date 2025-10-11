import json
from typing import List, Dict, Any
from metrics import compute_bleu, compute_rouge


def compute_metrics_for_annotations(auto_results: List[Dict[str, Any]], raw_data: List[Dict[str, Any]]):
    """
    计算自动标注的BLEU-4和ROUGE-1/2/L分数，并打印结果。
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
    print(f"自动标注 BLEU-4: {avg_bleu:.4f}")
    print(f"自动标注 ROUGE-1: {avg_rouge_1:.4f}  ROUGE-2: {avg_rouge_2:.4f}  ROUGE-L: {avg_rouge_l:.4f}  (共{total}条)")


def export_annotation_results(results: List[Dict[str, Any]], raw_data: List[Dict[str, Any]], output_path: str = "final_annotation_results.json"):
    """
    合并标注结果和原始数据, 并导出为JSON文件。
    :param results: 标注结果列表, 每项为dict, 需包含id字段
    :param raw_data: 原始数据列表, 每项为dict, 需包含id/question/context/answer等
    :param output_path: 导出文件路径
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
    print(f"已导出到 {output_path}")
