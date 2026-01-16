from typing import List, Dict, Any, Tuple, Callable
from abc import ABC, abstractmethod
from tqdm import tqdm
import json
import os

from utils import export_annotation_results
from scripts.select_best_route import select_best


class BaseRouter(ABC):
    """Abstract router interface for scoring candidate LLMs."""
    @property
    def if_train(self):
        """Whether this router requires training."""
        raise NotImplementedError()
    
    @abstractmethod
    def build_from_annotations(self, out_dir: str, **kwargs):
        """Build internal index from annotated data."""
        raise NotImplementedError()

    @abstractmethod
    def score(self, sample: str, candidate_llms: List[str]) -> List[Dict[str, Any]]:
        """Return list of {'model': name, 'score': float} for the candidates."""
        raise NotImplementedError()

    def choose_best(self, sample: str, candidate_llms: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
        scored = self.score(sample, candidate_llms)
        best = max(scored, key=lambda x: x.get('score', 0.0))
        return best['model'], scored
    
    # Main function entry point
    def route(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """see if the router needs training, if so, train it"""
        if self.if_train and not self.ready:
            self.cold_start(
                raw_dataset=dataset.to_list() if hasattr(dataset, 'to_list') else dataset,
                annotator=getattr(self, 'annotator', None),
                candidate_llms=getattr(self, 'candidate_llms', None),
                export_fn=export_annotation_results,
                select_best_fn=select_best,
                final_anno_file="best_router.json"
            )

        routed = []
        total = len(dataset) if hasattr(dataset, '__len__') else None
        progress_cb = getattr(self, 'progress_cb', None)
        for i, d in enumerate(tqdm(dataset)):
            # If a router is provided (e.g., MLPRouter), use it to score candidates
            scores = self.score(d.get('text', ''), self.candidate_llms)
            # scores -> list of {model, score}
            best = max(scores, key=lambda x: x.get('score', 0.0))
            chosen = best.get('model')
            routed.append({**d, 'route': chosen, 'route_scores': scores})
            # report progress if callback provided
            try:
                if progress_cb is not None and total is not None:
                    try:
                        progress_cb(i + 1, int(total), {'phase': 'route'})
                    except Exception:
                        pass
            except Exception:
                pass
        return routed

    def cold_start(self,
                   raw_dataset: List[Dict[str, Any]],
                   annotator,
                   candidate_llms: List[str],
                   export_fn: Callable,
                   select_best_fn: Callable,
                   final_anno_file: str = "best_router.json") -> List[Dict[str, Any]]:
        """Perform cold-start training by warm-up annotating a small budget.

        This will annotate the first `train_budget` samples with each candidate LLM,
        export the per-LLM annotation files using `export_fn`, select the best
        combined annotation using `select_best_fn`, then call
        `build_from_annotations` with the selected annotations.
        """
        train_budget = getattr(self, 'train_budget', None)
        if train_budget is None:
            raise AttributeError("Router missing attribute 'train_budget' required for cold_start")

        anno_file_list = []
        # If a final annotation file already exists, use it directly to build router
        if final_anno_file and os.path.exists(final_anno_file):
            with open(final_anno_file, encoding='utf-8') as f:
                annotations = json.load(f)
            # build index / internal structures from annotations
            self.build_from_annotations(annotations, out_dir="./")
            self.ready = True
            return annotations

        # If per-LLM warmup annotation files already exist for all candidates,
        # run the selection function to produce the final_anno_file and use it.
        per_llm_files = [f"{llm}_annos.json" for llm in candidate_llms]
        if per_llm_files and all(os.path.exists(p) for p in per_llm_files):
            try:
                select_best_fn(per_llm_files, out_path=final_anno_file)
                with open(final_anno_file, encoding='utf-8') as f:
                    annotations = json.load(f)
                self.build_from_annotations(annotations, out_dir="./")
                self.ready = True
                return annotations
            except Exception:
                # fall through to performing warm-up annotations if selection fails
                pass
        # print("Step 0: router cold-start")
        for llm in candidate_llms:
            print(f" - Warm-up annotating with {llm}")
            # pass progress callback if annotator exposes one
            progress_cb = getattr(annotator, 'progress_cb', None)
            annotations = annotator.annotate_batch(raw_dataset[:train_budget], assigned_llm=llm, progress_cb=progress_cb)
            export_fn(annotations, raw_dataset[:train_budget], output_path=f"{llm}_annos.json")
            anno_file_list.append(f"{llm}_annos.json")

        select_best_fn(anno_file_list, out_path=final_anno_file)
        with open(final_anno_file, encoding='utf-8') as f:
            annotations = json.load(f)

        # build index / internal structures from annotations
        self.build_from_annotations(annotations, out_dir="./")
        self.ready = True
