from typing import List, Dict, Any
from recommendation import ActiveLearningFilter, Refiner
from annotation import Annotator
from task import QATask
from load_squad import download_squad, load_squad_to_qa_list
from utils import export_annotation_results


# ==============================
# 主流程
# ==============================

class HumanLLMAnnotationSystem:
    def __init__(self, candidate_llms, task=QATask(), llm_mode="local", api_config=None):
        self.filter = ActiveLearningFilter(method="alps", budget=1000, batch_size=50)
        self.refiner = Refiner(candidate_llms, budget=100, llm_mode=llm_mode, api_config=api_config)
        self.annotator = Annotator(candidate_llms, llm_mode=llm_mode, api_config=api_config, task=task)

    def run(self, raw_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        print("Step 1: Active Learning 采样...")
        sampled_data = self.filter.select(raw_dataset)

        print("Step 2: LLM 精排 + 路由...")
        refined = self.refiner.refine_and_route(sampled_data)

        print("Step 3: LLM 标注 + 人工兜底...")
        annotated = self.annotator.annotate_batch(refined)

        # 导出人工复审池
        self.annotator.human_review_queue.export()

        return annotated


if __name__ == "__main__":
    # 下载并加载SQuAD v1.1数据集
    download_squad()
    raw_data = load_squad_to_qa_list(max_samples=10000)
    task = QATask()

    candidate_llms = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]
    system = HumanLLMAnnotationSystem(candidate_llms, task)
    results = system.run(raw_data)

    print(f"\n最终得到 {len(results)} 条标注结果")

    # 只导出自动标注通过的结果（needs_human=False）
    auto_results = [r for r in results if not r.get("needs_human", False)]
    export_annotation_results(auto_results, raw_data, output_path="final_annotation_results.json")
