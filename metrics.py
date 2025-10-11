# BLEU and ROUGE for Chinese/English
from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def compute_bleu(reference: str, hypothesis: str) -> float:
    """计算单句BLEU-4分数"""
    ref = [list(reference)]
    hyp = list(hypothesis)
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref, hyp, smoothing_function=smoothie)

def compute_rouge(reference: str, hypothesis: str) -> dict:
    """计算ROUGE-1/2/L分数"""
    rouge = Rouge()
    try:
        scores = rouge.get_scores(hypothesis, reference)[0]
    except Exception:
        scores = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}
    return scores
