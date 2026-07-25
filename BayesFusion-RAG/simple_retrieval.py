#!/usr/bin/env python3
"""
简化的 RAG 命令行检索工具
专注于从已处理的 JSON 数据中直接检索
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import re

class SimpleJSONRetriever:
    """简单的 JSON 数据检索器"""
    
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
                import re
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

def main():
    """主函数"""
    import sys

    # 检查命令行参数
    interactive_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == '--no-interactive':
        interactive_mode = False

    # 确定要搜索的 JSON 文件
    json_file = "./data/processed/GOVPUB-D101-PURL-LPS36916_fa1756dd.json"

    if not os.path.exists(json_file):
        print(f"❌ 找不到文件: {json_file}")
        print("请确保文件路径正确")
        return

    try:
        # 创建检索器
        retriever = SimpleJSONRetriever(json_file)
        retriever.load()

        # 5个预设问题
        questions = [
            "MANDATORY REPLACEMENT PARTS LIST",
            "HOW TO USE TORQUE TABLE",
            "REMOVAL AND INSTALLATION OF ENGINE- AND TRANSMISSION-RELATED COMPONENTS",
            "SCREW THREAD INSERTS (ONE-PIECE TYPE)",
            "QUICK GUIDE TO TROUBLESHOOTING"
        ]

        print("\n" + "="*80)
        print("🔍 开始检索预设问题...")
        print("="*80)

        for i, question in enumerate(questions, 1):
            print(f"\n❓ 问题 {i}: {question}")
            print("-" * 60)

            # 搜索相关内容
            results = retriever.search(question, top_k=10)

            if results:
                print(f"📋 找到 {len(results)} 个相关结果:")
                for j, result in enumerate(results, 1):
                    print(f"\n--- 结果 {j} (分数: {result.get('score', 0)}) ---")
                    print(f"文档: {result.get('document', '未知')}")
                    print(f"页面: {result.get('page', '未知')}")
                    print(f"标题: {result.get('title', '无标题')[:100]}...")
                    print(f"内容: {result.get('content', '无内容')[:300]}...")
            else:
                print("⚠️ 未找到相关结果")

            print("="*60)

        print("\n🎉 预设问题检索完成！")

        # 交互模式
        if interactive_mode:
            print("\n🔄 进入交互模式 (输入 'quit' 退出):")
            while True:
                try:
                    user_input = input("\n请输入问题: ").strip()
                    if user_input.lower() in ['quit', 'exit', '退出']:
                        break

                    if not user_input:
                        continue

                    print(f"\n❓ 问题: {user_input}")
                    print("-" * 40)

                    results = retriever.search(user_input, top_k=10)

                    if results:
                        print(f"📋 找到 {len(results)} 个相关结果:")
                        for j, result in enumerate(results, 1):
                            print(f"\n{'='*80}")
                            print(f"🔍 结果 {j} | 匹配分数: {result.get('score', 0):.2f} | 匹配度: {result.get('match_ratio', 0):.1%}")
                            print(f"📄 文档: {result.get('document', '未知')}")
                            print(f"📖 页面: {result.get('page', '未知')}")
                            if result.get('matched_words'):
                                print(f"🎯 匹配关键词: {', '.join(result.get('matched_words', []))}")
                            print(f"📝 标题: {result.get('title', '无标题')}")
                            print(f"📄 内容:")
                            content = result.get('content', '无内容')
                            # 显示完整内容，但如果太长则分段显示
                            if len(content) > 500:
                                print(f"   {content[:500]}...")
                                print(f"   [内容较长，共 {len(content)} 字符，显示前500字符]")
                                # 询问是否显示完整内容
                                show_full = input("\n   💡 是否显示完整内容？(y/n): ").strip().lower()
                                if show_full in ['y', 'yes', '是']:
                                    print(f"\n   📄 完整内容:\n   {content}")
                            else:
                                print(f"   {content}")
                            print(f"{'='*80}")
                    else:
                        print("⚠️ 未找到相关结果，请尝试其他关键词")

                except KeyboardInterrupt:
                    print("\n👋 再见！")
                    break
                except EOFError:
                    print("\n👋 检测到输入结束，退出程序！")
                    break
                except Exception as e:
                    print(f"❌ 处理出错: {e}")
        else:
            print("\n💡 提示: 使用 'python simple_retrieval.py' 进入交互模式")

    except Exception as e:
        print(f"❌ 初始化失败: {e}")

if __name__ == '__main__':
    main()