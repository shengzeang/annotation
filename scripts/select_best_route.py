"""Best router selection utility.

Selects the best-performing LLM route from a list of per-LLM annotation files
based on aggregate confidence scores.
"""

import json
import os
from typing import List


def select_best(anno_file_list: List[str], out_path: str = "best_route.json") -> str:
    """Select the annotation file whose samples have the highest mean confidence.

    Parameters
    ----------
    anno_file_list:
        Paths to per-LLM annotation JSON files.  Each file is expected to
        contain a list of dicts with at least a ``"confidence"`` field.
    out_path:
        Where to write (copy) the winning annotation file.

    Returns
    -------
    str
        Path of the winning annotation file.
    """
    if not anno_file_list:
        return out_path

    best_path = None
    best_score = float("-inf")

    for path in anno_file_list:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not data:
                continue
            mean_conf = sum(
                float(item.get("confidence", 0.0)) for item in data
            ) / len(data)
            if mean_conf > best_score:
                best_score = mean_conf
                best_path = path
        except Exception:
            continue

    if best_path is None:
        best_path = anno_file_list[0]

    # Copy the winning file to out_path.
    try:
        with open(best_path, "r", encoding="utf-8") as src, \
             open(out_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
    except Exception:
        pass

    return best_path
