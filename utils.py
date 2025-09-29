import json
from typing import List, Dict, Any

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
