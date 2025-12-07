import os
import sys

# Load the adapter module directly to avoid importing the package-level
# `filters.__init__` which pulls heavy dependencies (e.g. transformers).
repo_root = os.path.dirname(os.path.dirname(__file__))
mod_path = os.path.join(repo_root, "filters", "dataflow_adapter.py")
import importlib.util
spec = importlib.util.spec_from_file_location("dataflow_adapter", mod_path)
dataflow_adapter = importlib.util.module_from_spec(spec)
sys.path.insert(0, repo_root)
spec.loader.exec_module(dataflow_adapter)
DataFlowOperatorFilterAdapter = dataflow_adapter.DataFlowOperatorFilterAdapter


class PassThroughOperator:
    def __init__(self, **kwargs):
        pass

    def run(self, storage, input_key="text", output_key=None):
        # Read whatever the storage provides and write it back unchanged
        data = storage.read()
        storage.write(data)


def main():
    adapter = DataFlowOperatorFilterAdapter(operator_class=PassThroughOperator, budget=10)
    dataset = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
    out = adapter.filter(dataset)
    print("Filtered results:")
    for item in out:
        print(item)


if __name__ == "__main__":
    main()
