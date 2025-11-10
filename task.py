"""
task.py: 任务类型抽象基类
"""
from abc import ABC, abstractmethod
import unicodedata, re
from nltk.tokenize import word_tokenize, sentence_tokenize
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


class ClassificationTask(Task):
    """分类任务实现"""
    def get_prompt(self, sample: Dict[str, Any], rag_examples=None) -> str:
        rag_str = ""
        if rag_examples:
            rag_str = "\nHere are some similar classification examples from the knowledge base to help you classify:\n"
            for ex in rag_examples:
                rag_str += f"Text: {ex.get('text','')}\nCategory: {ex.get('annotation','')}\n"
        prompt = (
            f"Classify the following text into one of the predefined categories.\n"
            f"Also output a confidence score (between 0.0 and 1.0) for your answer, representing how confident you are in your answer.\n"
            f"Your answer should be one of the categories listed below.\n"
            f"Output format: Category: <category> Confidence: <score>\n"
            f"Categories: {sample.get('Categories', '')}\n"
            f"Text: {sample.get('Text','')}\n"
            f"{rag_str}"
            f"Answer:"
        )
        return prompt

    def parse_output(self, output: str) -> Dict[str, Any]:
        category, conf = "unknown", None
        if "Confidence" in output:
            try:
                parts = output.split("Confidence")
                category = parts[0].split(":")[-1].strip().replace(",","") #extract category
                conf = float(parts[1].split("\n")[0].replace(":", "").strip()) #extract confidence
            except:
                pass
        if conf == None:
            return {"annotation": category}
        else:
            return {"annotation": category, "confidence": conf}
        

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
        

class Translation(Task):
    """翻译任务实现: English to Chinese"""
    def __init__(self, target_language="Chinese"):
        self.target_language = target_language

    def get_dictionary_hints(self, en_text, dictionary: Dict[str, str]) -> str:
        hints = []
        for token in en_text.split():
            if token in dictionary:
                hints.append(f"{token}: {dictionary[token]}")
        if hints:
            return "\n".join(hints)
        else:
            return ""
        
    def get_prompt(self, sample: Dict[str, Any], dictionary = None, rag_examples=None) -> str:
        text = sample.get("text", "")
        prompt = (
            f"You are a English to Chinese Translator. Translate the following text into {self.target_language}.\n"
            f"Also output a confidence score (between 0.0 and 1.0) for your answer, representing how confident you are in your answer.\n"
            f"Text: {text}\n"
            f"Dictionary hints: {self.get_dictionary_hints(text, dictionary) if dictionary else 'None'}\n"
            f"Output format: Translation: <translated text> Confidence: <score>\n"
            f"Translation:"
        )
        return prompt

    def parse_output(self, output: str) -> Dict[str, Any]:
        translation, conf = "unknown", None
        if "Confidence" in output:
            try:
                parts = output.split("Confidence")
                translation = parts[0].split("Translation:")[-1].strip().replace(",", "")
                conf = float(parts[1].split("\n")[0].replace(":", "").strip())
                return {"annotation": translation, "confidence": conf}
            except:
                pass
        if conf == None:
            return {"annotation": translation}
        else:
            return {"annotation": translation, "confidence": conf}
        

class TextSummarization(Task):
    """文本摘要任务实现"""
    def __init__(self, max_len=150):
        self.summary_type = "extractive"  # or "abstractive"
        self.max_len = max_len
    
    def get_prompt(self, sample: Dict[str, Any], rag_examples=None) -> str:
        rag_str = ""
        if rag_examples:
            rag_str = "\nHere are some similar summarization examples from the knowledge base to help you summarize:\n"
            for ex in rag_examples:
                rag_str += f"Text: {ex.get('text','')}\nSummary: {ex.get('summary','')}\n"
        prompt = (
            f"Summarize the following text into a concise summary.\n"
            f"The summary should not exceed {self.max_len} words.\n"
            f"Also output a confidence score (between 0.0 and 1.0) for your answer, representing how confident you are in your answer.\n"
            f"Output format: Summary: <summary text> Confidence: <score>\n"
            f"Text: {sample.get("text", "")}\n"
            f"{rag_str}"
            f"Summary:"
        )
        return prompt

    def parse_output(self, output: str) -> Dict[str, Any]:
        summary, conf = "unknown", None
        if "Confidence" in output:
            try:
                parts = output.split("Confidence")
                summary = parts[0].split("Summary:")[-1].strip()
                conf = float(parts[1].split("\n")[0].replace(":", "").strip())
            except:
                pass
        if conf == None:
            return {"annotation": summary}
        else:
            return {"annotation": summary, "confidence": conf}
