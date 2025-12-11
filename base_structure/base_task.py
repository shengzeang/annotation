from abc import ABC, abstractmethod
from typing import Dict, Any


class Task(ABC):
    """任务类型抽象基类"""
    @abstractmethod
    #生成发送给LLM的prompt
    def get_prompt(self, sample: Dict[str, Any], rag_examples=None) -> str:
        """根据样本和可选RAG检索结果生成prompt"""
        pass
    #解析LLM的返回结果
    @abstractmethod
    def parse_output(self, output: str) -> Dict[str, Any]:
        """解析LLM输出, 返回结构化标注结果"""
        pass