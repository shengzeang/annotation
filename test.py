from typing import List, Dict, Any
import logging
import json
from annotation import Annotator
from filters import ActiveLearningFilter, LLMNaiveFilter
from routers import KNNRouter, CascadeRouter
from tasks.qa import QATask
from qa_data import SquadDataset
from utils import export_annotation_results
from misc.llm_provider import LocalLLM, APILLM
from scripts.select_best_route import select_best

# Default LLMs for the full annotation pipeline (cheapest → most capable).
DEFAULT_CANDIDATE_LLMS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]

# ==============================
# main process
# ==============================

class HumanLLMAnnotationSystem:
    def __init__(self, candidate_llms=None, task=QATask(), llm_mode="local", api_config=None):
        if candidate_llms is None:
            candidate_llms = DEFAULT_CANDIDATE_LLMS
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
        # Pick the middle candidate as the naive pre-filter LLM so the choice
        # scales gracefully when candidate_llms is customised.
        _filter_llm_key = candidate_llms[len(candidate_llms) // 2]
        self.filter_2 = LLMNaiveFilter(self.llm_dict[_filter_llm_key], budget=500)
        # create annotator first and pass it to the router which requires it
        self.annotator = Annotator(self.candidate_llms, self.llm_dict, task=task)
        self.router = KNNRouter(self.annotator, candidate_llms, encoder_name="sentence-transformers/all-MiniLM-L6-v2", k=5, train_budget=100)
        # self.router = CascadeRouter(judge_llm=self.llm_dict[candidate_llms[-1]], candidate_llm=candidate_llms, llm_dict=self.llm_dict, threshold=0.7)


    def run(self, raw_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # ensure train_budget is available regardless of router implementation
        train_budget = getattr(self.router, 'train_budget', 0)
        if self.router.if_train:
            # perform router cold-start using centralized method on router
            self.router.cold_start(raw_dataset,
                                  annotator=self.annotator,
                                  candidate_llms=self.candidate_llms,
                                  export_fn=export_annotation_results,
                                  select_best_fn=select_best,
                                  final_anno_file="best_knn.json")

        logger = logging.getLogger(__name__)
        logger.info("Step 1: stream filter")
        filtered_data = self.filter_1.filter(raw_dataset[train_budget:])
        filtered_data = self.filter_2.filter(filtered_data)
        logger.info("Step 2: LLM route")
        routed = self.router.route(filtered_data)
        logger.info("Step 3: LLM annotation")
        annotated = self.annotator.annotate_batch(routed)

        # 导出人工复审池
        self.annotator.human_review_queue.export()
        return annotated


if __name__ == "__main__":
    raw_data = SquadDataset.from_url(save_path="squad_train.json", max_samples=10000, skip_initial=500, shuffle_seed=42)
    task = QATask()

    system = HumanLLMAnnotationSystem(task=task)
    results = system.run(raw_data)

    logger = logging.getLogger(__name__)
    logger.info("\n最终得到 %d 条标注结果", len(results))

    # 只导出自动标注通过的结果（needs_human=False）
    auto_results = [r for r in results if not r.get("needs_human", False)]
    export_annotation_results(auto_results, raw_data, output_path="our_anno_new.json")

    # 计算自动标注BLEU和ROUGE分数
    from utils import compute_metrics_for_annotations
    compute_metrics_for_annotations(auto_results, raw_data)
