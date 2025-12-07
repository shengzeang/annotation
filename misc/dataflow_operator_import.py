from dataflow.utils.registry import OPERATOR_REGISTRY
from typing import List, Dict, Any, Type
from dataflow.core import OperatorABC
import importlib
import os
#from dataflow.operators.general_text.filter.blocklist_filter import BlocklistFilter
FILTER_PACKAGE = "dataflow.operators.general_text.filter"

def load_all_dataflow_operators() -> Dict[str, Type[OperatorABC]]:
    """
    Dynamically eval filter operators in general_text.filter 
    Returns: { operator_name: operator_class } (registry["Filter"] = <class PerspectiveFilter>)
    """
    registry = {}

    # locate directory
    module = importlib.import_module(FILTER_PACKAGE)
    filter_dir = os.path.dirname(module.__file__)

    for filename in os.listdir(filter_dir):
        if filename.endswith("_filter.py") and filename != "__init__.py":
            module_name = filename[:-3] #remove '.py'
            module_obj = importlib.import_module(f"{FILTER_PACKAGE}.{module_name}")
            #find class names ending in "Filter"
            for attr_name in dir(module_obj):
                if attr_name.endswith("Filter"):
                    cls = getattr(module_obj, attr_name)
                    if isinstance(cls, type) and issubclass(cls, OperatorABC):
                        registry[attr_name.lower()] = cls #store key as lowercase

    return registry

DATAFLOW_FILTER_REGISTRY = load_all_dataflow_operators()

def get_operator(name: str) -> Type[OperatorABC]:
    return DATAFLOW_FILTER_REGISTRY.get(name.lower())
