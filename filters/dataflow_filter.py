from typing import List, Dict, Any, Optional, Type
from base_structure.base_filter import BaseFilter
from base_structure.dataset import Dataset, DatasetStorage
import os


class DataFlowFilter(BaseFilter):
    """
    Adapter that lets repository filters use operators from a DataFlow-like
    operator registry, or accept a custom operator class directly.

    Usage:
      - Pass `operator_name` to load via `dataflow_operator_import.get_operator`
        (when that module and `dataflow` operators are available).
      - Or pass `operator_class` directly (recommended if you want to avoid
        the dynamic import dependency).
    """

    def __init__(
        self,
        operator_name: Optional[str] = None,
        operator_class: Optional[Type] = None,
        budget: int = 200,
        operator_args: Optional[Dict[str, Any]] = None,
    ):
        if not operator_name and operator_class is None:
            raise ValueError("Either operator_name or operator_class must be provided")

        self.operator_name = operator_name
        self.operator_class = operator_class
        self.budget = budget
        self.operator_args = operator_args or {}

        # If operator name is provided, try to resolve via local import
        if self.operator_class is None and self.operator_name is not None:
            try:
                from ..misc import dataflow_operator_import as dfi

                op = dfi.get_operator(self.operator_name)
                if op is None:
                    raise ValueError(f"Operator '{self.operator_name}' not found in registry")
                self.operator_class = op
            except Exception as e:
                raise ImportError(
                    f"Unable to load operator '{self.operator_name}': {e}.\n"
                    "Ensure `dataflow_operator_import` and the `dataflow` package are available"
                )

        self.cache_path = "./cache"
        os.makedirs(self.cache_path, exist_ok=True)

    def _run_operator_on_text(self, text: str) -> float:
        """
        Run the configured operator on a single text value.
        Expectation: operator writes a dataframe with the kept rows.
        Returns 1.0 for kept, 0.0 for filtered-out. On errors returns 0.0.
        """
        # Use the project's Dataset-backed storage
        df = [{"text": text}]
        storage = DatasetStorage()
        storage.write(df)

        try:
            op = self.operator_class(**self.operator_args)
            # Try common signatures used by known operators
            try:
                op.run(storage, input_key="text", output_key="label")
            except TypeError:
                try:
                    op.run(storage, input_key="text")
                except Exception:
                    # give operator one more shot without kwargs
                    op.run(storage)

            out_df = storage.read()
            if out_df is None:
                return 0.0

            # If operator returned a Dataset, use its length
            try:
                if isinstance(out_df, Dataset):
                    kept = len(out_df)
                elif isinstance(out_df, list):
                    kept = len(out_df)
                elif isinstance(out_df, dict):
                    kept = 1
                elif hasattr(out_df, "__len__"):
                    kept = len(out_df)
                else:
                    kept = 1 if out_df else 0
            except Exception:
                kept = 0

            return 1.0 if kept == 1 else 0.0
        except Exception:
            return 0.0

    def filter(self, raw_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored_items = []

        name = self.operator_name or getattr(self.operator_class, "__name__", "operator")

        for d in raw_dataset:
            text = d.get("text", "")
            score = self._run_operator_on_text(text)
            d_with_score = {**d, f"{name}_score": score}
            scored_items.append(d_with_score)

        scored_items.sort(key=lambda x: x[f"{name}_score"], reverse=True)
        return scored_items[: self.budget]
