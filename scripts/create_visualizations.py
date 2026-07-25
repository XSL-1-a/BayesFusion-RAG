#!/usr/bin/env python3
"""
论文可视化脚本
1. 检索案例分析对比图
2. 实体门控效果可视化
3. 方法性能对比图
"""

import os
import sys
import json
import csv
import random
import re
from pathlib import Path
from typing import List, Dict
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

MAINTNET_DIR = project_root / "data" / "maintnet"
OUTPUT_DIR = project_root / "results" / "maintnet"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 无头模式
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available, will generate text-based visualizations")


def load_v4_results():
    """加载 v4 实验结果"""
    path = OUTPUT_DIR / "maintnet_results_v4.json"
    with open(path, 'r') as f:
        return json.load(f)


def create_performance_comparison_chart(results: Dict):
    """创建方法性能对比柱状图"""
    if not HAS_MPL:
        return create_text_performance_table(results)

    methods = ['BM25', 'Dense', 'Hybrid', 'RRF', 'HyDE', 'Ours-HMR', 'Ours-Full']
    metrics = ['mrr', 'ndcg', 'recall']
    metric_labels = ['MRR', 'NDCG@5', 'Recall@5']

    # 跨域平均
    avg_scores = {m: {} for m in methods}
    domains = list(results['main_results'].keys())
    for method in methods:
        for metric in metrics:
            scores = [results['main_results'][d][method][metric] for d in domains]
            avg_scores[method][metric] = np.mean(scores)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(len(methods))
    width = 0.7

    colors = ['#95a5a6', '#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = [avg_scores[m][metric] for m in methods]

        bars = ax.bar(x, values, width, color=colors)

        # 高亮我们的方法
        bars[-1].set_color('#e74c3c')
        bars[-2].set_color('#e74c3c')
        bars[-1].set_edgecolor('black')
        bars[-2].set_edgecolor('black')
        bars[-1].set_linewidth(2)
        bars[-2].set_linewidth(2)

        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylim(0, 1)

        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Cross-Domain Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIGURE_DIR / 'performance_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {FIGURE_DIR / 'performance_comparison.png'}")


def create_text_performance_table(results: Dict):
    """文本格式的性能表格"""
    methods = ['BM25', 'Dense', 'Hybrid', 'RRF', 'HyDE', 'Ours-HMR', 'Ours-Full']
    metrics = ['mrr', 'ndcg', 'recall']

    domains = list(results['main_results'].keys())
    print("\n  【跨域平均性能】")
    print(f"  {'Method':<12} {'MRR':>8} {'NDCG@5':>8} {'Recall':>8}")
    print("  " + "-" * 40)

    for method in methods:
        scores = {}
        for metric in metrics:
            vals = [results['main_results'][d][method][metric] for d in domains]
            scores[metric] = np.mean(vals)
        marker = " ★" if method.startswith('Ours') else ""
        print(f"  {method:<12} {scores['mrr']:>8.3f} {scores['ndcg']:>8.3f} {scores['recall']:>8.3f}{marker}")


def create_domain_heatmap(results: Dict):
    """创建域-方法热力图"""
    if not HAS_MPL:
        return

    methods = ['BM25', 'Dense', 'Hybrid', 'RRF', 'HyDE', 'Ours-HMR', 'Ours-Full']
    domains = ['aviation', 'automotive', 'facility']
    domain_labels = ['Aviation', 'Automotive', 'Facility']

    # MRR 矩阵
    mrr_matrix = np.array([
        [results['main_results'][d][m]['mrr'] for m in methods]
        for d in domains
    ])

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mrr_matrix, cmap='RdYlGn', aspect='auto', vmin=0.2, vmax=1.0)

    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticklabels(domain_labels)

    # 添加数值
    for i in range(len(domains)):
        for j in range(len(methods)):
            text = ax.text(j, i, f'{mrr_matrix[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=10)

    ax.set_title('MRR by Domain and Method', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='MRR')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'domain_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {FIGURE_DIR / 'domain_heatmap.png'}")


def create_ablation_chart(results: Dict):
    """创建消融实验图"""
    if not HAS_MPL:
        return create_text_ablation_table(results)

    configs = ['w/o Both\n(Dense)', 'w/o HMR\n(Dense+Entity)', 'w/o Entity\n(HMR only)', 'Full\n(HMR+Entity)']
    config_keys = ['w/o Both (Dense)', 'w/o HMR (Dense+Entity)', 'w/o Entity (HMR only)', 'Full (HMR+Entity)']

    aviation_mrr = [results['ablation']['aviation'][k]['mrr'] for k in config_keys]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#bdc3c7', '#95a5a6', '#3498db', '#e74c3c']
    bars = ax.bar(configs, aviation_mrr, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('MRR')
    ax.set_ylim(0.5, 0.65)
    ax.set_title('Ablation Study on Aviation Dataset', fontsize=14, fontweight='bold')

    # 添加数值和提升标注
    for i, (bar, val) in enumerate(zip(bars, aviation_mrr)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 添加提升箭头
    base = aviation_mrr[0]
    for i in range(1, len(aviation_mrr)):
        delta = aviation_mrr[i] - base
        if delta > 0:
            ax.annotate('', xy=(i, aviation_mrr[i]), xytext=(0, base),
                       arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'ablation_study.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {FIGURE_DIR / 'ablation_study.png'}")


def create_text_ablation_table(results: Dict):
    """文本格式的消融实验表格"""
    configs = ['w/o Both (Dense)', 'w/o HMR (Dense+Entity)', 'w/o Entity (HMR only)', 'Full (HMR+Entity)']
    print("\n  【消融实验 - Aviation】")
    print(f"  {'Configuration':<30} {'MRR':>8} {'Δ':>8}")
    print("  " + "-" * 50)

    base = results['ablation']['aviation']['w/o Both (Dense)']['mrr']
    for config in configs:
        mrr = results['ablation']['aviation'][config]['mrr']
        delta = mrr - base
        print(f"  {config:<30} {mrr:>8.3f} {delta:>+8.3f}")


def create_entity_gating_analysis():
    """创建实体门控效果分析"""
    # 门控统计 (从 v4 实验日志)
    gating_stats = {
        'Aviation': {'applied': 232, 'total': 1233, 'rate': 18.8},
        'Automotive': {'applied': 0, 'total': 38, 'rate': 0},
        'Facility': {'applied': 0, 'total': 40, 'rate': 0},
    }

    if not HAS_MPL:
        print("\n  【实体门控应用率】")
        for domain, stats in gating_stats.items():
            print(f"  {domain}: {stats['applied']}/{stats['total']} ({stats['rate']:.1f}%)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图: 门控应用率
    ax1 = axes[0]
    domains = list(gating_stats.keys())
    rates = [gating_stats[d]['rate'] for d in domains]
    colors = ['#3498db', '#95a5a6', '#95a5a6']

    bars = ax1.bar(domains, rates, color=colors, edgecolor='black')
    ax1.set_ylabel('Entity Gate Application Rate (%)')
    ax1.set_title('Entity Gating Rate by Domain', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 25)

    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)

    # 右图: 门控条件说明
    ax2 = axes[1]
    ax2.axis('off')

    gate_text = """
    Entity Gating Conditions:

    Gate 1: Semantic Completeness
    → Query must contain BOTH component AND symptom
    → Example: "magneto leaking" ✓
    → Example: "HVAC issue" ✗ (no symptom)

    Gate 2: Score Discrimination (std > 0.15)
    → Entity scores must show variance

    Gate 3: Max-Mean Gap (> 0.20)
    → Top candidate must stand out

    Effect:
    • Aviation: 18.8% queries have reliable entity signal
    • Facility: 0% (category-only queries lack specificity)
    • Automotive: 0% (similar pattern)

    → Prevents entity noise from hurting performance
    """
    ax2.text(0.1, 0.95, gate_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'entity_gating_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {FIGURE_DIR / 'entity_gating_analysis.png'}")


def create_hyde_comparison_chart():
    """创建 HyDE vs 其他方法的对比图"""
    if not HAS_MPL:
        return

    # 数据
    methods = ['Dense', 'HyDE', 'Ours-Full']
    aviation_mrr = [0.559, 0.511, 0.590]
    facility_mrr = [0.562, 0.463, 0.722]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, aviation_mrr, width, label='Aviation', color='#3498db')
    bars2 = ax.bar(x + width/2, facility_mrr, width, label='Facility', color='#e74c3c')

    ax.set_ylabel('MRR')
    ax.set_title('HyDE Performance Degradation vs Our Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 0.85)

    # 添加数值
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

    # 添加下降箭头
    ax.annotate('', xy=(1, 0.511), xytext=(0, 0.559),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.5, 0.48, '−8.6%', ha='center', fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'hyde_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {FIGURE_DIR / 'hyde_comparison.png'}")


def create_retrieval_case_study():
    """创建检索案例分析"""
    case_study = {
        'query': "What maintenance action is needed when gasket shows leaking?",
        'ground_truth': "REPLACED GASKETS",
        'methods': {
            'BM25': {
                'top1': "Problem: EXHAUST PIPE LOOSE\nAction: TIGHTENED CLAMPS",
                'rank': 5,
                'relevant': False,
            },
            'Dense': {
                'top1': "Problem: #2 INTAKE GASKET LEAKING\nAction: REPLACED GASKET",
                'rank': 1,
                'relevant': True,
            },
            'HyDE': {
                'top1': "Problem: OIL FILTER HOUSING GASKET SEEPING\nAction: REPLACED GASKET AND TORQUED",
                'rank': 2,
                'relevant': True,
            },
            'Ours-Full': {
                'top1': "Problem: #1 INTAKE GASKET LEAKING\nAction: REPLACED GASKET",
                'rank': 1,
                'relevant': True,
            },
        }
    }

    print("\n  【检索案例分析】")
    print(f"\n  Query: {case_study['query']}")
    print(f"  Ground Truth: {case_study['ground_truth']}")
    print("\n  Top-1 Retrieved Results:")
    print("  " + "-" * 60)

    for method, result in case_study['methods'].items():
        status = "✓" if result['relevant'] else "✗"
        print(f"\n  [{method}] {status}")
        print(f"    {result['top1'][:80]}...")
        print(f"    Rank of relevant: {result['rank']}")

    # 保存为 JSON
    with open(FIGURE_DIR / 'case_study.json', 'w') as f:
        json.dump(case_study, f, indent=2)


def main():
    print("=" * 70)
    print("  论文可视化生成")
    print("=" * 70)

    # 加载结果
    print("\n[1/6] 加载实验结果...")
    results = load_v4_results()

    # 生成各种图表
    print("\n[2/6] 生成性能对比图...")
    create_performance_comparison_chart(results)

    print("\n[3/6] 生成域热力图...")
    create_domain_heatmap(results)

    print("\n[4/6] 生成消融实验图...")
    create_ablation_chart(results)

    print("\n[5/6] 生成实体门控分析图...")
    create_entity_gating_analysis()

    print("\n[6/6] 生成 HyDE 对比图和案例分析...")
    create_hyde_comparison_chart()
    create_retrieval_case_study()

    print("\n" + "=" * 70)
    print(f"  所有图表已保存至: {FIGURE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
