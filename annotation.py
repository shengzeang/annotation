import random
from typing import List, Dict, Any
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from llm_provider import LocalLLM, APILLM
from task import Task, QATask


# ==============================
# Annotation 模块
# ==============================
class HumanReviewQueue:
    """人工复审池"""
    def __init__(self):
        self.queue = []

    def add(self, sample: Dict[str, Any]):
        self.queue.append(sample)

    def export(self, filepath: str = "human_review.json"):
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.queue, f, ensure_ascii=False, indent=2)
        print(f"Human review queue exported to {filepath}")


class Annotator:
    """使用 Qwen 进行标注, LLM调用抽象化"""
    def __init__(self, candidate_llms, confidence_threshold: float = 0.7, llm_mode: str = "local", api_config: dict = None,
                 rag_method: str = "bm25", kb_path: str = "knowledge_base.json", task: Task = None):
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

        self.confidence_threshold = confidence_threshold
        self.human_review_queue = HumanReviewQueue()
        self.kb_path = kb_path
        self.rag_method = rag_method.lower() if rag_method else "bm25"
        # 加载本地知识库
        self.knowledge_base = self._load_knowledge_base()
        self.task = task or QATask()

    def _load_knowledge_base(self):
        import os, json
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    kb = json.load(f)
                print(f"Loaded knowledge base from {self.kb_path}, size={len(kb)}")
                return kb
            except Exception as e:
                print(f"Failed to load knowledge base: {e}")
        return []

    def _save_knowledge_base(self):
        import json
        with open(self.kb_path, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
        # print(f"Knowledge base updated: {self.kb_path}, size={len(self.knowledge_base)}")

    def _rag_retrieve(self, question: str, topk: int = 3):
        """支持BM25和TF-IDF两种RAG检索方式, 用户可选, 默认BM25"""
        if not self.knowledge_base:
            return []
        questions = [item.get("question", "") for item in self.knowledge_base]
        if self.rag_method == "bm25":
            tokenized_corpus = [q.lower().split() for q in questions]
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(question.lower().split())
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
            return [self.knowledge_base[i] for i in top_indices if scores[i] > 0]
        elif self.rag_method == "tfidf":
            tfidf = TfidfVectorizer().fit(questions + [question])
            q_vecs = tfidf.transform(questions)
            query_vec = tfidf.transform([question])
            sims = cosine_similarity(query_vec, q_vecs)[0]
            top_indices = sims.argsort()[::-1][:topk]
            return [self.knowledge_base[i] for i in top_indices if sims[i] > 0]
        else:
            # fallback: 词重叠
            scored = []
            for item in self.knowledge_base:
                q2 = item.get("question", "")
                overlap = len(set(question.lower().split()) & set(q2.lower().split()))
                scored.append((overlap, item))
            scored.sort(reverse=True, key=lambda x: x[0])
            return [x[1] for x in scored[:topk] if x[0] > 0]

    def annotate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # RAG 检索相似历史标注
        rag_examples = self._rag_retrieve(sample.get("question", ""), topk=3)
        # 通过Task对象生成prompt
        prompt = self.task.get_prompt(sample, rag_examples)
        llm = sample.get('route')
        if llm not in self.candidate_llms:
            best_llm = self.candidate_llms[0]
            for candidate in self.candidate_llms:
                if candidate in str(llm):
                    best_llm = candidate
            llm = best_llm
        output = self.llm_dict[llm].generate(prompt, max_new_tokens=50)

        # 通过Task对象解析LLM输出
        parsed = self.task.parse_output(output)
        annotation = parsed.get("annotation", "unknown")
        conf = parsed.get("confidence", random.random())

        result = {**sample, "route": llm, "annotation": annotation, "confidence": conf}
        if conf < self.confidence_threshold:
            result["needs_human"] = True
            self.human_review_queue.add(result)
        else:
            result["needs_human"] = False
            # 标注通过的样本加入知识库并保存
            self.knowledge_base.append(result)
            self._save_knowledge_base()
        return result

    def annotate_batch(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.annotate(d) for d in tqdm(dataset)]
