"""
RAGAS-style evaluation using LLM-as-judge.
Evaluates: Faithfulness, Context Relevance, Context Recall, Answer Relevance.
"""
import json
import re
import numpy as np
from src.agent.llm_client import LLMClient
from src.semantic.embedding_model import EmbeddingModel

def _cosine(a, b) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)

class RagasEvaluator:
    def __init__(self):
        self.llm = LLMClient()
        self.emb = EmbeddingModel()

    def _call_llm(self, prompt: str) -> str:
        # LLMClient generate takes system_prompt, user_prompt
        return self.llm.generate("You are a strict output JSON evaluation system.", prompt)

    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        if not answer.strip() or not contexts:
            return 0.0

        context_text = "\n\n".join(contexts)
        prompt = f"""Context:
{context_text}

Answer:
{answer}

Extract each individual factual statement from the answer.
For each statement decide if it can be directly inferred from the context.

Respond with ONLY valid JSON — no explanation, no markdown:
{{"statements": [{{"statement": "...", "supported": true}}]}}"""

        try:
            data = _parse_json(self._call_llm(prompt))
            stmts = data.get("statements", [])
            if not stmts:
                return 0.0
            return round(sum(1 for s in stmts if s.get("supported")) / len(stmts), 4)
        except Exception as e:
            return 0.0

    def context_relevance(self, query: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0

        context_text = "\n\n".join(contexts)
        prompt = f"""Query: {query}

Context:
{context_text}

Count the total number of sentences in the context.
Then count how many of those sentences are useful for answering the query.

Respond with ONLY valid JSON — no explanation, no markdown:
{{"relevant_count": 3, "total_count": 5}}"""

        try:
            data = _parse_json(self._call_llm(prompt))
            relevant = int(data.get("relevant_count", 0))
            total = int(data.get("total_count", 1))
            if total == 0:
                return 0.0
            return round(min(1.0, relevant / total), 4)
        except Exception:
            return 0.0

    def context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        if not ground_truth.strip() or not contexts:
            return 0.0

        context_text = "\n\n".join(contexts)
        prompt = f"""Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context_text}

Break the ground truth into individual factual claims.
For each claim decide if it can be found in or inferred from the context.

Respond with ONLY valid JSON — no explanation, no markdown:
{{"claims": [{{"claim": "...", "in_context": true}}]}}"""

        try:
            data = _parse_json(self._call_llm(prompt))
            claims = data.get("claims", [])
            if not claims:
                return 0.0
            return round(sum(1 for c in claims if c.get("in_context")) / len(claims), 4)
        except Exception:
            return 0.0

    def answer_relevance(self, query: str, answer: str) -> float:
        if not answer.strip():
            return 0.0

        prompt = f"""Answer: {answer}

Generate exactly 3 questions that this answer would best respond to.

Respond with ONLY valid JSON — no explanation, no markdown:
{{"questions": ["q1", "q2", "q3"]}}"""

        try:
            data = _parse_json(self._call_llm(prompt))
            questions = data.get("questions", [])
            if not questions:
                return 0.0

            q_vec = self.emb.encode([query])[0]
            gen_vecs = self.emb.encode(questions)
            similarities = [_cosine(q_vec, gv) for gv in gen_vecs]
            return round(max(0.0, min(1.0, sum(similarities) / len(similarities))), 4)
        except Exception:
            return 0.0

    def evaluate_sample(self, query: str, ground_truth: str, answer: str, contexts: list[str]):
        return {
            "query": query,
            "faithfulness": self.faithfulness(answer, contexts),
            "context_relevance": self.context_relevance(query, contexts),
            "context_recall": self.context_recall(ground_truth, contexts),
            "answer_relevance": self.answer_relevance(query, answer),
        }

if __name__ == "__main__":
    print("Initializing RAGAS Evaluator...")
    evaluator = RagasEvaluator()
    
    sample_query = "What is the capital of India?"
    sample_gt = "The capital of India is New Delhi."
    sample_answer = "New Delhi is the capital."
    sample_ctx = ["New Delhi is the capital of India, replacing Calcutta in 1911."]
    
    print("Evaluating sample...")
    result = evaluator.evaluate_sample(sample_query, sample_gt, sample_answer, sample_ctx)
    print(json.dumps(result, indent=2))
