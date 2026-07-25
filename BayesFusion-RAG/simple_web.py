#!/usr/bin/env python3
"""
极简Web界面 - 专门用于图片测试
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from demo.app_pdf_chat import PDFChatSystem
from PIL import Image

# 全局系统实例
system = None

def init_system():
    global system
    if system and system.initialized:
        return "✅ 系统已就绪"

    system = PDFChatSystem(fast_mode=True)

    class SimpleProgress:
        def __call__(self, progress, desc=""):
            pass  # 静默初始化

    try:
        system.initialize(SimpleProgress())
        return "✅ 系统初始化成功！可以开始测试图片"
    except Exception as e:
        return f"❌ 初始化失败: {e}"

def test_image(image, question):
    global system
    if not system or not system.initialized:
        return "请先点击初始化按钮"

    if image is None:
        return "请上传图片"

    try:
        if question.strip():
            # 图片+问题
            result = system._process_image(image, question)
        else:
            # 仅识别图片来源
            result = system.find_image_source(image)

        return result
    except Exception as e:
        return f"处理失败: {e}"

def test_text(question):
    global system
    if not system or not system.initialized:
        return "请先点击初始化按钮"

    if not question.strip():
        return "请输入问题"

    try:
        message = {"text": question, "files": []}
        _, history, _ = system.chat(message, [], "所有文档")

        if history and len(history) >= 2:
            return history[-1].get('content', '无回答')
        return "无回答"
    except Exception as e:
        return f"查询失败: {e}"

# 创建极简界面
with gr.Blocks(title="PDF RAG 测试") as demo:
    gr.Markdown("# 📚 PDF RAG 图片测试")

    # 初始化
    init_btn = gr.Button("🚀 初始化系统", variant="primary")
    status = gr.Textbox(label="状态", value="点击初始化按钮开始")

    # 图片测试
    gr.Markdown("## 🖼️ 图片测试")
    with gr.Row():
        image = gr.Image(label="上传图片", type="pil")
        with gr.Column():
            question = gr.Textbox(label="问题（可选）", placeholder="这张图片显示什么？")
            img_btn = gr.Button("分析图片")

    img_result = gr.Textbox(label="图片分析结果", lines=10)

    # 文本测试
    gr.Markdown("## 💬 文本测试")
    with gr.Row():
        text_input = gr.Textbox(label="输入问题", placeholder="F-2 MANDATORY REPLACEMENT PARTS LIST")
        text_btn = gr.Button("查询")

    text_result = gr.Textbox(label="查询结果", lines=8)

    # 快速测试
    with gr.Row():
        test1 = gr.Button("测试1: F-2零件清单")
        test2 = gr.Button("测试2: 发动机拆装")
        test3 = gr.Button("测试3: 故障排除")

    # 事件绑定
    init_btn.click(init_system, outputs=[status])
    img_btn.click(test_image, inputs=[image, question], outputs=[img_result])
    text_btn.click(test_text, inputs=[text_input], outputs=[text_result])
    text_input.submit(test_text, inputs=[text_input], outputs=[text_result])

    # 快速测试事件
    test1.click(lambda: "F-2 MANDATORY REPLACEMENT PARTS LIST", outputs=[text_input])
    test2.click(lambda: "REMOVAL AND INSTALLATION OF ENGINE AND TRANSMISSION RELATED COMPONENTS", outputs=[text_input])
    test3.click(lambda: "QUICK GUIDE TO TROUBLESHOOTING", outputs=[text_input])

if __name__ == '__main__':
    print("启动极简PDF RAG测试系统...")
    demo.launch(
        server_name="127.0.0.1",  # 仅本地访问
        server_port=7875,
        share=False,
        show_error=True,
        quiet=True
    )
