"""
task.py: 任务类型抽象基类与QA任务实现
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class Task(ABC):
    """任务类型抽象基类"""
    @abstractmethod
    def get_prompt(self, sample: Dict[str, Any], rag_examples=None) -> str:
        """根据样本和可选RAG检索结果生成prompt"""
        pass

    @abstractmethod
    def parse_output(self, output: str) -> Dict[str, Any]:
        """解析LLM输出, 返回结构化标注结果"""
        pass

class QATask(Task):
    """QA任务实现"""
    def get_prompt(self, sample: Dict[str, Any], rag_examples=None) -> str:
        rag_str = ""
        if rag_examples:
            rag_str = "\nHere are some similar QA pairs from the knowledge base to help you answer:\n"
            for ex in rag_examples:
                rag_str += f"Q: {ex.get('question','')}\nA: {ex.get('annotation','')}\n"
        prompt = (
            f"Given the following question, please answer it as accurately as possible.\n"
            f"Also output a confidence score (between 0.0 and 1.0) for your answer, representing how confident you are in your answer.\n"
            f"Output format: Answer: <your answer> Confidence: <score>\n"
            f"Question: {sample.get('question', sample.get('text',''))}\n"
            f"Context: {sample.get('context', '')}\n"
            f"{rag_str}"
            f"Answer:"
        )
        return prompt

    def parse_output(self, output: str) -> Dict[str, Any]:
        annotation, conf = "unknown", None
        if "Confidence" in output:
            try:
                parts = output.split("Confidence")
                annotation = parts[0].split(":")[-1].strip().replace(",", "")
                conf = float(parts[1].split("\n")[0].replace(":", "").strip())
            except:
                pass
        if conf == None:
            return {"annotation": annotation}
        else:
            return {"annotation": annotation, "confidence": conf}
