from typing import List, Dict, Any
from annotation import Annotator
from filtering import ActiveLearningFilter, LLMNaiveFilter
from routing import KNNRouter
from task import QATask
from misc.load_squad import download_squad, load_squad_to_qa_list
from utils import export_annotation_results
from misc.llm_provider import LocalLLM, APILLM


# ==============================
# 主流程
# ==============================

class HumanLLMAnnotationSystem:
    def __init__(self, candidate_llms, task=QATask(), llm_mode="local", api_config=None):
        self.candidate_llms = candidate_llms
        self.llm_mode = llm_mode
        self.llm_dict = {}
        if llm_mode == "local":
            for llm in candidate_llms:
                self.llm_dict[llm] = LocalLLM(llm)
        elif llm_mode == "api":
            for llm in candidate_llms:
                conf = api_config.get(llm, {}) if api_config else {}
                self.llm_dict[llm] = APILLM(conf.get("api_url", ""), conf.get("api_key"), conf.get("extra_headers"))
        else:
            raise ValueError(f"Unknown llm_mode: {llm_mode}")


        self.filter_1 = ActiveLearningFilter(method="alps", budget=1000, batch_size=50)
        self.filter_2 = LLMNaiveFilter(self.llm_dict["Qwen/Qwen2.5-7B-Instruct"], budget=500)
        self.router = KNNRouter(candidate_llms=candidate_llms, encoder_name="sentence-transformers/all-MiniLM-L6-v2", k=5, ann_path="our_anno_knn.json")
        self.annotator = Annotator(self.candidate_llms, self.llm_dict, task=task)

    def run(self, raw_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        print("Step 1: stream filter")
        filtered_data = self.filter_1.filter(raw_dataset)
        filtered_data = self.filter_2.filter(filtered_data)

        print("Step 2: LLM route")
        routed = self.router.route(filtered_data)

        print("Step 3: LLM annotation")
        annotated = self.annotator.annotate_batch(routed)

        # 导出人工复审池
        self.annotator.human_review_queue.export()

        return annotated


if __name__ == "__main__":
    # 下载并加载SQuAD v1.1数据集
    download_squad()
    raw_data = load_squad_to_qa_list(max_samples=10000)
    task = QATask()

    candidate_llms = ["Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",]
            #"Qwen/Qwen2.5-32B-Instruct"]
    # candidate_llms = ["Qwen/Qwen2.5-32B-Instruct"]
    system = HumanLLMAnnotationSystem(candidate_llms, task)
    results = system.run(raw_data)

    print(f"\n最终得到 {len(results)} 条标注结果")

    # 只导出自动标注通过的结果（needs_human=False）
    auto_results = [r for r in results if not r.get("needs_human", False)]
    export_annotation_results(auto_results, raw_data, output_path="our_anno_new.json")

    # 计算自动标注BLEU和ROUGE分数
    from utils import compute_metrics_for_annotations
    compute_metrics_for_annotations(auto_results, raw_data)
