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


def load_annotations(path: Path | str) -> Dict[str, Dict[str, Any]]:
    """Load annotation file and return mapping id -> record dict.

    Expected record has at least 'id' and a text field (one of: 'annotation', 'answer', 'text', 'prediction').
    """
    # Accept either a Path or a string path
    p = Path(path) if not isinstance(path, Path) else path
    data = json.loads(p.read_text(encoding='utf-8'))
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
    # download and load SQuAD via SquadDataset helper (preserves previous skip behavior)
    try:
        from datasets import SquadDataset
    except Exception:
        # import error (possibly circular) — fallback: load JSON directly if present
        import json, os
        fp = 'squad_train.json'
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # mimic Dataset.to_list structure
            ds = type('Tmp', (), {'to_list': lambda self: data})()
        else:
            raise
    else:
        ds = SquadDataset.from_url(save_path="squad_train.json", max_samples=500, skip_initial=500)
    data = ds.to_list()
    out: Dict[str, str] = {}
    for rec in data:
        sid = rec.get('id') or rec.get('idx') or rec.get('q_id')
        if sid is None:
            sid = rec.get('question') or rec.get('text') or rec.get('context')
        ref = rec.get('annotation') or rec.get('answer') or rec.get('reference') or rec.get('gold') or rec.get('text')
        if ref is None:
            ref = ''
        out[str(sid)] = ref
    return out


def select_best(input_files: List[Path] | List[str], out_path: Path | str):
    # normalize inputs to Path objects
    input_paths = [Path(p) for p in input_files]
    # load inputs
    anns = [load_annotations(p) for p in input_paths]
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

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {len(results)} best-route records to {out_path}')


def main():
    p = argparse.ArgumentParser(description='Select best route per sample by BLEU across multiple annotation files')
    p.add_argument('--inputs', '-i', nargs='+', required=True, help='Annotation JSON files to compare')
    p.add_argument('--out', '-o', required=True, help='Output JSON file to write best routes')
    args = p.parse_args()

    input_files = args.inputs
    for f in input_files:
        if not Path(f).exists():
            raise FileNotFoundError(f)

    out_path = args.out
    select_best(input_files, out_path)


if __name__ == '__main__':
    main()
