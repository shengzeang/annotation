from abc import ABC, abstractmethod
import unicodedata, re
from typing import Dict, Any
from ..base_structure.base_task import Task


class NERTask(Task): 
    """命名实体识别任务实现"""
    def __init__(self, entity_types=None, language="zh"):
        self.entity_types = ["PERSON", "ORG", "GPE", "LOC", "DATE","CARDINAL","LANGUAGE"] if entity_types is None else entity_types #后期调用
        self.language = language
    
    def pre_process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """预处理样本"""
        text = sample.get("text", "")
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[ \t]+", " ", text)
        sample["processed_text"] = text 
        return sample 

    def get_prompt(self, sample: Dict[str, Any], rag_examples=None) -> str:
        sample = self.pre_process(sample)
        text = sample.get("processed_text", sample.get("text", "")) #优先使用processed text
        rag_str = ""
        if rag_examples:
            rag_str = "\nHere are some similar NER examples from the knowledge base to help you identify entities:\n"
            for ex in rag_examples:
                rag_str += f"Text: {ex.get('text','')}\nEntities: {ex.get('annotation','')}\n"
        
        prompt = (
            f"Identify and extract named entities from the following text.\n"
            f"Also output a confidence score (between 0.0 and 1.0) for your answer, representing how confident you are in your answer.\n"
            f"Only use the following entity types: {', '.join(self.entity_types)}\n"
            f"Output format: Entities: entity|type1, entity2|type2,... Confidence: <score>\n"
            f"Text: {text}\n"
            f"{rag_str}"
            f"Answer:"
        )
        return prompt

    #example output: annotation: John|PERSON, PKU|ORG ,Confidence: 0.95
    def parse_output(self, output: str) -> Dict[str, Any]:
        annotation, conf = "unknown", None
        if "Confidence" in output:
            try:
                parts = output.split("Confidence")
                conf = float(parts[1].split("\n")[0].replace(":", "").strip()) #extract confidence
                annotation = parts[0].split("Entities:")[-1].strip() #extract entities
            except:
                pass
        if conf == None:
            return {"annotation": annotation}
        else:
            return {"annotation": annotation, "confidence": conf}