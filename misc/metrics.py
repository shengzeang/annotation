# BLEU and ROUGE for Chinese/English
from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU-4 score"""
    ref = [list(reference)]
    hyp = list(hypothesis)
    smoothie = SmoothingFunction().method1
    return float(sentence_bleu(ref, hyp, smoothing_function=smoothie))

def compute_rouge(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE-1/2/L scores"""
    rouge = Rouge()
    try:
        scores = rouge.get_scores(hypothesis, reference)[0]
    except Exception:
        scores = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}
    return scores
