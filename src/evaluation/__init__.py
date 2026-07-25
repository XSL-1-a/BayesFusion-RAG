"""
评估框架模块
- 答案质量评估 (Answer Correctness, Relevancy)
- 检索质量评估 (Context Relevancy, MRR, nDCG)
- 忠实度评估 (Faithfulness, Hallucination Rate)
- 实体抽取评估 (Entity F1, Source Attribution)
- 可追溯性评估 (Citation Precision/Recall)
"""

from .metrics import MetricsCalculator
from .llm_judge import LLMJudge
from .human_eval import HumanEvaluationInterface
from .experiment_runner import ExperimentRunner
from .result_analyzer import ResultAnalyzer

__all__ = [
    "MetricsCalculator",
    "LLMJudge",
    "HumanEvaluationInterface",
    "ExperimentRunner",
    "ResultAnalyzer"
]
