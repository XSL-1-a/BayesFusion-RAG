"""
人工评估接口
支持人工标注和评估
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.logger import LoggerMixin
from ..utils.data_types import TraceableAnswer


class HumanEvaluationInterface(LoggerMixin):
    """
    人工评估接口
    生成评估任务并收集人工标注
    """

    def __init__(self, output_dir: str = "./experiments/human_eval"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_evaluation_task(
        self,
        answers: List[TraceableAnswer],
        task_name: str,
        evaluation_dimensions: Optional[List[str]] = None
    ) -> str:
        """
        创建人工评估任务

        Args:
            answers: 待评估的答案列表
            task_name: 任务名称
            evaluation_dimensions: 评估维度

        Returns:
            任务文件路径
        """
        if evaluation_dimensions is None:
            evaluation_dimensions = [
                "answer_correctness",
                "answer_completeness",
                "citation_accuracy",
                "readability"
            ]

        # 构建评估任务
        task = {
            "task_name": task_name,
            "created_at": datetime.now().isoformat(),
            "dimensions": evaluation_dimensions,
            "items": []
        }

        for i, answer in enumerate(answers):
            item = {
                "id": i + 1,
                "query": answer.query,
                "answer": answer.answer,
                "citations": [
                    {
                        "id": c.citation_id,
                        "claim": c.claim,
                        "source": f"{c.source.document}, Page {c.source.page}"
                    }
                    for c in answer.citations
                ],
                "sources_summary": [
                    {
                        "id": j + 1,
                        "document": s.source.document,
                        "page": s.source.page,
                        "modality": s.modality.value
                    }
                    for j, s in enumerate(answer.sources)
                ],
                "evaluation": {dim: None for dim in evaluation_dimensions}
            }
            task["items"].append(item)

        # 保存任务
        filepath = self.output_dir / f"{task_name}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(task, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Created evaluation task: {filepath}")

        return str(filepath)

    def load_evaluation_results(self, task_file: str) -> Dict[str, Any]:
        """
        加载人工评估结果

        Args:
            task_file: 任务文件路径

        Returns:
            评估结果
        """
        with open(task_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def aggregate_human_scores(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        聚合人工评估分数

        Args:
            results: 评估结果

        Returns:
            聚合分数
        """
        dimensions = results.get("dimensions", [])
        items = results.get("items", [])

        aggregated = {dim: [] for dim in dimensions}

        for item in items:
            evaluation = item.get("evaluation", {})
            for dim in dimensions:
                score = evaluation.get(dim)
                if score is not None:
                    aggregated[dim].append(score)

        return {
            dim: sum(scores) / len(scores) if scores else 0
            for dim, scores in aggregated.items()
        }

    def generate_annotation_guide(self) -> str:
        """生成标注指南"""
        guide = """
# 人工评估标注指南

## 评估维度

### 1. 答案正确性 (answer_correctness)
- 5: 完全正确，所有信息准确
- 4: 大部分正确，有轻微不准确
- 3: 部分正确，有明显错误
- 2: 大部分错误
- 1: 完全错误

### 2. 答案完整性 (answer_completeness)
- 5: 完全回答了问题的所有方面
- 4: 回答了大部分方面
- 3: 回答了部分方面
- 2: 回答很不完整
- 1: 几乎没有回答问题

### 3. 引用准确性 (citation_accuracy)
- 5: 所有引用都正确指向相关来源
- 4: 大部分引用正确
- 3: 约一半引用正确
- 2: 大部分引用错误
- 1: 引用完全错误

### 4. 可读性 (readability)
- 5: 非常清晰易读
- 4: 比较清晰
- 3: 一般
- 2: 比较难读
- 1: 非常难读

## 标注说明
1. 请仔细阅读问题和答案
2. 检查引用是否正确
3. 对每个维度给出1-5的分数
4. 如有需要，添加备注
"""
        return guide
