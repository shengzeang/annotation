"""Select best route per sample by BLEU score across multiple annotation files.

Usage examples:
  python scripts/select_best_route.py --inputs ann_qwen.json ann_llama.json --ref gold.json --out best_routes.json
  python scripts/select_best_route.py --inputs final_annotation_results.json other_ann.json --out best_routes.json

If --ref is omitted the script will use the first input file as the reference (warning will be printed).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from misc.metrics import compute_bleu
from misc.load_squad import download_squad, load_squad_to_qa_list


def load_annotations(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load annotation file and return mapping id -> record dict.

    Expected record has at least 'id' and a text field (one of: 'annotation', 'answer', 'text', 'prediction').
    """
    data = json.loads(path.read_text(encoding='utf-8'))
    out = {}
    for rec in data:
        sid = rec.get('id') or rec.get('idx') or rec.get('q_id')
        if sid is None:
            # try to construct an id from question/context
            sid = rec.get('question') or rec.get('text') or rec.get('context')
        question = rec.get('question', '')
        context = rec.get('context', '')
        annotation = rec.get('annotation')
        route = rec.get('route') or rec.get('model') or rec.get('llm')
        out[str(sid)] = {'raw': rec, 'question': question, 'context': context, 'annotation': annotation, 'route': route}
    return out


def load_reference() -> Dict[str, str]:
    download_squad()
    data = load_squad_to_qa_list(max_samples=500)
    out = {}
    for rec in data:
        sid = rec.get('id') or rec.get('idx') or rec.get('q_id')
        if sid is None:
            sid = rec.get('question') or rec.get('text') or rec.get('context')
        ref = rec.get('annotation') or rec.get('answer') or rec.get('reference') or rec.get('gold') or rec.get('text')
        if ref is None:
            ref = ''
        out[str(sid)] = ref
    return out


def select_best(input_files: List[Path], out_path: Path):
    # load inputs
    anns = [load_annotations(p) for p in input_files]
    # union of ids
    all_ids = set()
    for a in anns:
        all_ids.update(a.keys())

    ref_map = load_reference()

    results = []
    for sid in sorted(all_ids):
        ref = ref_map.get(sid, '')
        best_score = -1.0
        best_route = None
        question = ''
        context = ''
        for i, a in enumerate(anns):
            rec = a.get(sid)
            if rec is None:
                # missing in this file
                continue
            hyp = rec.get('annotation', '')
            try:
                score = float(compute_bleu(ref or '', hyp or ''))
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best_route = rec.get('route')
                best_text = hyp
                question = rec.get('question', '')
                context = rec.get('context', '')
        results.append({
            'id': sid,
            'question': question,
            'context': context,
            'route': best_route,
            'annotation': best_text,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {len(results)} best-route records to {out_path}')


def main():
    p = argparse.ArgumentParser(description='Select best route per sample by BLEU across multiple annotation files')
    p.add_argument('--inputs', '-i', nargs='+', required=True, help='Annotation JSON files to compare')
    p.add_argument('--out', '-o', required=True, help='Output JSON file to write best routes')
    args = p.parse_args()

    input_files = [Path(x) for x in args.inputs]
    for f in input_files:
        if not f.exists():
            raise FileNotFoundError(f)

    out_path = Path(args.out)
    select_best(input_files, out_path)


if __name__ == '__main__':
    main()
