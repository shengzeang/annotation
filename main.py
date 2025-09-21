from typing import List, Dict, Any

from .recommendation import ActiveLearningRanker, QwenRefiner
from .annotation import QwenAnnotator


# ==============================
# 主流程
# ==============================

class HumanLLMAnnotationSystem:
    def __init__(self, candidate_llms):
        self.ranker = ActiveLearningRanker()
        self.refiner = QwenRefiner(candidate_llms)
        self.annotator = QwenAnnotator(candidate_llms)

    def run(self, raw_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("Step 1: Active Learning 粗排...")
        ranked = self.ranker.rank(raw_dataset)

        print("Step 2: Qwen 精排 + 路由...")
        refined = self.refiner.refine_and_route(ranked)

        print("Step 3: Qwen 标注 + 人工兜底...")
        annotated = self.annotator.annotate_batch(refined)

        # 导出人工复审池
        self.annotator.human_review_queue.export()

        return annotated


if __name__ == "__main__":
    raw_data = [{"id": i, "text": f"这是第 {i} 条样本"} for i in range(200)]

    candidate_llms = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"]
    system = HumanLLMAnnotationSystem(candidate_llms)
    results = system.run(raw_data)

    print(f"\n最终得到 {len(results)} 条标注结果")
    print("前5条结果: ")
    for r in results[:5]:
        print(r)