"""
LLM评判模块
使用LLM评估答案质量
"""

from typing import Dict, Any, Optional, List

from ..utils.logger import LoggerMixin
from ..utils.llm_client import LLMClient
from ..utils.data_types import TraceableAnswer, RetrievalResult


class LLMJudge(LoggerMixin):
    """
    LLM评判器
    使用LLM评估RAG系统的各项质量指标
    """

    ANSWER_CORRECTNESS_PROMPT = """Evaluate the correctness of the generated answer compared to the reference answer.

Question: {question}

Generated Answer: {generated}

Reference Answer: {reference}

Evaluate on a scale of 0.0 to 1.0:
- 1.0: Completely correct, all key information matches
- 0.7-0.9: Mostly correct with minor differences
- 0.4-0.6: Partially correct, some key information missing or wrong
- 0.1-0.3: Mostly incorrect
- 0.0: Completely wrong or unrelated

Output JSON:
{{
    "score": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

    CONTEXT_RELEVANCY_PROMPT = """Evaluate how relevant the retrieved context is to the question.

Question: {question}

Retrieved Context:
{context}

Evaluate relevancy on a scale of 0.0 to 1.0:
- 1.0: Highly relevant, directly answers the question
- 0.7-0.9: Mostly relevant, contains useful information
- 0.4-0.6: Somewhat relevant, peripherally related
- 0.1-0.3: Barely relevant
- 0.0: Completely irrelevant

Output JSON:
{{
    "score": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

    FAITHFULNESS_PROMPT = """Evaluate if the answer is faithful to the provided sources.

Question: {question}

Sources:
{sources}

Generated Answer: {answer}

Evaluate faithfulness on a scale of 0.0 to 1.0:
- 1.0: Completely faithful, all claims supported by sources
- 0.7-0.9: Mostly faithful, minor unsupported details
- 0.4-0.6: Partially faithful, some unsupported claims
- 0.1-0.3: Mostly unfaithful
- 0.0: Contains fabricated information

Output JSON:
{{
    "score": 0.0-1.0,
    "unsupported_claims": ["list of claims not in sources"],
    "reasoning": "brief explanation"
}}"""

    ANSWER_RELEVANCY_PROMPT = """Evaluate if the answer is relevant and addresses the question.

Question: {question}

Generated Answer: {answer}

Evaluate relevancy on a scale of 0.0 to 1.0:
- 1.0: Directly and completely addresses the question
- 0.7-0.9: Mostly addresses the question
- 0.4-0.6: Partially addresses the question
- 0.1-0.3: Barely addresses the question
- 0.0: Does not address the question

Output JSON:
{{
    "score": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

    def __init__(self, model: Optional[str] = None, num_samples: int = 1):
        self.model = model
        self.num_samples = num_samples
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = LLMClient.get_client()
        return self._llm_client

    def judge_answer_correctness(
        self,
        question: str,
        generated: str,
        reference: str
    ) -> Dict[str, Any]:
        """评估答案正确性"""
        prompt = self.ANSWER_CORRECTNESS_PROMPT.format(
            question=question,
            generated=generated,
            reference=reference
        )

        return self._get_judgment(prompt)

    def judge_context_relevancy(
        self,
        question: str,
        contexts: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """评估上下文相关性"""
        context_text = "\n\n".join([
            f"[{i+1}] {c.content[:500]}..."
            for i, c in enumerate(contexts)
        ])

        prompt = self.CONTEXT_RELEVANCY_PROMPT.format(
            question=question,
            context=context_text
        )

        return self._get_judgment(prompt)

    def judge_faithfulness(
        self,
        question: str,
        sources: List[RetrievalResult],
        answer: str
    ) -> Dict[str, Any]:
        """评估答案忠实度"""
        sources_text = "\n\n".join([
            f"[Source {i+1}] {s.content[:500]}..."
            for i, s in enumerate(sources)
        ])

        prompt = self.FAITHFULNESS_PROMPT.format(
            question=question,
            sources=sources_text,
            answer=answer
        )

        return self._get_judgment(prompt)

    def judge_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> Dict[str, Any]:
        """评估答案相关性"""
        prompt = self.ANSWER_RELEVANCY_PROMPT.format(
            question=question,
            answer=answer
        )

        return self._get_judgment(prompt)

    def _get_judgment(self, prompt: str) -> Dict[str, Any]:
        """获取LLM评判结果"""
        try:
            scores = []
            for _ in range(self.num_samples):
                response = self.llm_client.generate_structured(prompt)
                if "score" in response:
                    scores.append(response["score"])

            if scores:
                avg_score = sum(scores) / len(scores)
                return {
                    "score": avg_score,
                    "scores": scores,
                    "reasoning": response.get("reasoning", "")
                }

            return {"score": 0.5, "error": "No valid scores"}

        except Exception as e:
            self.logger.error(f"LLM judgment failed: {e}")
            return {"score": 0.5, "error": str(e)}

    def judge_all(
        self,
        answer: TraceableAnswer,
        reference: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        评估所有指标

        Args:
            answer: TraceableAnswer对象
            reference: 参考答案（可选）

        Returns:
            所有评判结果
        """
        results = {}

        # 上下文相关性
        results["context_relevancy"] = self.judge_context_relevancy(
            answer.query,
            answer.sources
        )

        # 答案相关性
        results["answer_relevancy"] = self.judge_answer_relevancy(
            answer.query,
            answer.answer
        )

        # 忠实度
        results["faithfulness"] = self.judge_faithfulness(
            answer.query,
            answer.sources,
            answer.answer
        )

        # 答案正确性（需要参考答案）
        if reference:
            results["answer_correctness"] = self.judge_answer_correctness(
                answer.query,
                answer.answer,
                reference
            )

        return results
