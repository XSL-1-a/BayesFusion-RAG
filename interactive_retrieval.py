#!/usr/bin/env python3
"""
交互式 RAG 检索工具
支持用户自定义问题，提供完整的检索效果展示
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import re

class InteractiveRAGRetriever:
    """交互式 RAG 检索器"""

    def __init__(self, json_file: str):
        self.json_file = Path(json_file)
        self.data = None
        self.chunks = []
        self.documents = {}

    def load(self):
        """加载 JSON 数据"""
        print(f"📂 加载数据文件: {self.json_file}")

        if not self.json_file.exists():
            raise FileNotFoundError(f"文件不存在: {self.json_file}")

        with open(self.json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 提取文档和块信息
        if 'documents' in self.data:
            self.documents = self.data['documents']

        if 'chunks' in self.data:
            self.chunks = self.data['chunks']
        else:
            # 如果没有 chunks 字段，尝试从其他结构提取
            self.chunks = self._extract_chunks_from_data()

        print(f"✅ 加载完成: {len(self.documents)} 文档, {len(self.chunks)} 块")

    def _extract_chunks_from_data(self) -> List[Dict[str, Any]]:
        """从数据结构中提取文本块"""
        chunks = []

        # 尝试不同的数据结构
        if isinstance(self.data, list):
            for i, item in enumerate(self.data):
                if isinstance(item, dict):
                    if 'content' in item:
                        chunks.append({
                            'id': item.get('id', str(i)),
                            'content': item['content'],
                            'title': item.get('title', ''),
                            'page': item.get('page', 0),
                            'document': item.get('document', '')
                        })
                    elif 'text' in item:
                        chunks.append({
                            'id': item.get('id', str(i)),
                            'content': item['text'],
                            'title': item.get('title', ''),
                            'page': item.get('page', 0),
                            'document': item.get('document', '')
                        })

        return chunks

    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """增强的关键词搜索"""
        query_words = query.lower().split()
        results = []

        for chunk in self.chunks:
            content = chunk.get('content', '').lower()
            title = chunk.get('title', '').lower()
            document = chunk.get('document', '').lower()

            # 计算匹配分数
            score = 0
            matched_words = []

            for word in query_words:
                word_score = 0
                # 精确匹配
                if word in content:
                    word_score += content.count(word) * 1
                if word in title:
                    word_score += title.count(word) * 3  # 标题权重更高
                if word in document:
                    word_score += document.count(word) * 2

                # 部分匹配（包含该词的词语）
                pattern = re.compile(r'\b\w*' + re.escape(word) + r'\w*\b')
                partial_matches = len(pattern.findall(content))
                if partial_matches > 0:
                    word_score += partial_matches * 0.5

                if word_score > 0:
                    matched_words.append(word)
                    score += word_score

            # 计算匹配度（匹配的词数/总词数）
            match_ratio = len(matched_words) / len(query_words) if query_words else 0
            final_score = score * match_ratio

            if final_score >= min_score:
                chunk_copy = chunk.copy()
                chunk_copy['score'] = final_score
                chunk_copy['matched_words'] = matched_words
                chunk_copy['match_ratio'] = match_ratio
                results.append(chunk_copy)

        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def display_result(self, result: Dict[str, Any], index: int):
        """格式化显示单个搜索结果"""
        print(f"\n{'='*80}")
        print(f"🔍 结果 {index} | 匹配分数: {result.get('score', 0):.2f} | 匹配度: {result.get('match_ratio', 0):.1%}")
        print(f"📄 文档: {result.get('document', '未知')}")
        print(f"📖 页面: {result.get('page', '未知')}")

        if result.get('matched_words'):
            print(f"🎯 匹配关键词: {', '.join(result.get('matched_words', []))}")

        title = result.get('title', '无标题')
        if title and title != '无标题':
            print(f"📝 标题: {title}")

        content = result.get('content', '无内容')
        print(f"📄 内容:")

        # 智能内容显示
        if len(content) > 800:
            print(f"   {content[:800]}...")
            print(f"   [内容较长，共 {len(content)} 字符，显示前800字符]")

            try:
                show_full = input("\n   💡 是否显示完整内容？(y/n): ").strip().lower()
                if show_full in ['y', 'yes', '是', '1']:
                    print(f"\n   📄 完整内容:\n   {content}")
            except (EOFError, KeyboardInterrupt):
                pass
        else:
            print(f"   {content}")

        print(f"{'='*80}")

def main():
    """主函数"""
    # 确定要搜索的 JSON 文件
    json_file = "./data/processed/GOVPUB-D101-PURL-LPS36916_fa1756dd.json"

    if not os.path.exists(json_file):
        print(f"❌ 找不到文件: {json_file}")
        print("请确保文件路径正确")
        return

    try:
        # 创建检索器
        retriever = InteractiveRAGRetriever(json_file)
        retriever.load()

        print("\n" + "="*80)
        print("🎯 交互式 RAG 检索系统")
        print("="*80)
        print("💡 使用说明:")
        print("   - 输入任何问题进行搜索")
        print("   - 输入 'quit', 'exit', 'q' 或 '退出' 来退出")
        print("   - 输入 'demo' 查看预设问题演示")
        print("   - 使用 Ctrl+C 也可以退出")
        print("="*80)

        while True:
            try:
                user_input = input("\n🔍 请输入问题: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 再见！感谢使用！")
                    break

                if not user_input:
                    print("⚠️ 请输入有效的问题")
                    continue

                # 演示模式
                if user_input.lower() == 'demo':
                    demo_questions = [
                        "MANDATORY REPLACEMENT PARTS LIST",
                        "HOW TO USE TORQUE TABLE",
                        "REMOVAL AND INSTALLATION OF ENGINE",
                        "SCREW THREAD INSERTS",
                        "TROUBLESHOOTING"
                    ]

                    print("\n🎬 演示模式 - 预设问题:")
                    for i, q in enumerate(demo_questions, 1):
                        print(f"   {i}. {q}")

                    try:
                        choice = input("\n选择问题编号 (1-5) 或直接回车跳过: ").strip()
                        if choice.isdigit() and 1 <= int(choice) <= 5:
                            user_input = demo_questions[int(choice) - 1]
                        else:
                            continue
                    except (ValueError, EOFError, KeyboardInterrupt):
                        continue

                print(f"\n❓ 搜索问题: {user_input}")
                print("-" * 60)

                # 执行搜索
                results = retriever.search(user_input, top_k=5)

                if results:
                    print(f"\n📋 找到 {len(results)} 个相关结果:")

                    for j, result in enumerate(results, 1):
                        retriever.display_result(result, j)

                        # 每显示2个结果后询问是否继续
                        if j % 2 == 0 and j < len(results):
                            try:
                                continue_show = input(f"\n   📖 还有 {len(results) - j} 个结果，是否继续显示？(y/n): ").strip().lower()
                                if continue_show not in ['y', 'yes', '是', '1']:
                                    break
                            except (EOFError, KeyboardInterrupt):
                                break

                    print(f"\n✅ 搜索完成！共显示了 {min(j, len(results))} 个结果")
                else:
                    print("⚠️ 未找到相关结果")
                    print("💡 建议:")
                    print("   - 尝试使用不同的关键词")
                    print("   - 使用更简单或更具体的词语")
                    print("   - 检查拼写是否正确")

            except KeyboardInterrupt:
                print("\n\n👋 再见！感谢使用！")
                break
            except EOFError:
                print("\n\n👋 再见！感谢使用！")
                break
            except Exception as e:
                print(f"❌ 处理出错: {e}")
                print("💡 请重试或检查输入")

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("💡 请检查数据文件是否存在且格式正确")

if __name__ == '__main__':
    main()
