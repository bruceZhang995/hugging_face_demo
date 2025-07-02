# 安装核心库（最新版）
# !pip install transformers gradio torch
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # 禁用符号链接警告
import gradio as gr
from transformers import pipeline
from opencc import OpenCC  # 用于简繁转换
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, AutoModelForCausalLM,AutoTokenizer

### pipeline支持的task
"""
"audio-classification": will return a [`AudioClassificationPipeline`].
"automatic-speech-recognition": will return a [`AutomaticSpeechRecognitionPipeline`].
"conversational": will return a [`ConversationalPipeline`].
"depth-estimation": will return a [`DepthEstimationPipeline`].
"document-question-answering": will return a [`DocumentQuestionAnsweringPipeline`].
"feature-extraction": will return a [`FeatureExtractionPipeline`].
"fill-mask": will return a [`FillMaskPipeline`]:.
"image-classification": will return a [`ImageClassificationPipeline`].
"image-feature-extraction": will return an [`ImageFeatureExtractionPipeline`].
"image-segmentation": will return a [`ImageSegmentationPipeline`].
"image-to-image": will return a [`ImageToImagePipeline`].
"image-to-text": will return a [`ImageToTextPipeline`].
"mask-generation": will return a [`MaskGenerationPipeline`].
"object-detection": will return a [`ObjectDetectionPipeline`].
"question-answering": will return a [`QuestionAnsweringPipeline`].
"summarization": will return a [`SummarizationPipeline`].
"table-question-answering": will return a [`TableQuestionAnsweringPipeline`].
"text2text-generation": will return a [`Text2TextGenerationPipeline`].
"text-classification"` (alias `"sentiment-analysis"` available): will return a [`TextClassificationPipeline`].
"text-generation": will return a [`TextGenerationPipeline`]:.
"text-to-audio"` (alias `"text-to-speech"` available): will return a [`TextToAudioPipeline`]:.
"token-classification"` (alias `"ner"` available): will return a [`TokenClassificationPipeline`].
"translation": will return a [`TranslationPipeline`].
"translation_xx_to_yy": will return a [`TranslationPipeline`].
"video-classification": will return a [`VideoClassificationPipeline`].
"visual-question-answering": will return a [`VisualQuestionAnsweringPipeline`].
"zero-shot-classification": will return a [`ZeroShotClassificationPipeline`].
"zero-shot-image-classification": will return a [`ZeroShotImageClassificationPipeline`].
"zero-shot-audio-classification": will return a [`ZeroShotAudioClassificationPipeline`].
"zero-shot-object-detection": will return a [`ZeroShotObjectDetectionPipeline`].
"""

# 简繁转换器
cc = OpenCC('t2s')  # 繁体转简体

# 创建生成器（添加更安全的模型加载）
try:
    model_path = r"./models/gpt2-chinese-cluecorpussmall"

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # generator = pipeline('text-generation', model='gpt2', device=0)
    generator = pipeline('text-generation', model=model,tokenizer=tokenizer, device=0)
    print("加载成功!")

except:
    # 备用方案：使用更小模型
    generator = pipeline('text-generation', model='distilgpt2', device=0)

    try:
        # 备用中文模型
        generator = pipeline('text-generation',
                            model='IDEA-CCNL/Wenzhong-GPT2-110M',
                            device=0)
    except:
        # 最后尝试英文模型
        generator = pipeline('text-generation', model='gpt2', device=0)

def generate_text(prompt, max_length=150, temperature=0.7, top_p=0.9, repetition_penalty=1.5):
    """
        temperature         0.5 - 0.9       值越高越天马行空，值越低越保守
        top_p               0.85 - 0.95     只考虑概率累积达阈值的词汇
        repetition_penalty  1.2 - 2.0       有效避免重复短语
        max_length          80 - 200        根据需求调整输出长度
    """
    params = {
        "max_length": int(max_length),  # 长度（50-200之间）
        "num_return_sequences":1,       # 固定生成1个结果
        "temperature": float(temperature),      # 控制随机性：0.3-0.9（值越低越保守）
        "top_k": 50,            # 核采样（过滤低概率词）; 固定值
        "top_p": float(top_p),  # 限制候选词数量
        "repetition_penalty": float(repetition_penalty), # 抑制重复（>1.0生效）
        "do_sample":True,       # 必须开启采样模式
    }
    try:
        result = generator(prompt, **params)[0]['generated_text']
        # 简繁转换（确保输出简体中文）
        simplified_result = cc.convert(result)
        return simplified_result
    except Exception as e:
        return f"生成失败: {str(e)}"
    return result

def main():

    # 使用Blocks创建更复杂的界面
    with gr.Blocks(title="AI文本生成实验室") as demo:
        gr.Markdown("## 🎮 文本生成参数调节台")

        with gr.Row():
            # 左侧：输入和控制区
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="输入提示",
                    placeholder="在这里输入你的创意开头...",
                    lines=3
                )

                with gr.Accordion("高级参数", open=False):
                    max_length_slider = gr.Slider(
                        minimum=50, maximum=300, value=100, step=10,
                        label="生成长度", interactive=True
                    )

                    temperature_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                        label="随机性 (temperature)", interactive=True
                    )

                    top_p_slider = gr.Slider(
                        minimum=0.5, maximum=1.0, value=0.9, step=0.05,
                        label="多样性 (top-p)", interactive=True
                    )

                    repetition_penalty_slider = gr.Slider(
                        minimum=1.0, maximum=2.0, value=1.2, step=0.1,
                        label="防重复惩罚", interactive=True
                    )

                generate_btn = gr.Button("生成文本", variant="primary")

            # 右侧：输出区
            with gr.Column():
                output_text = gr.Textbox(
                    label="生成结果（简体中文）",
                    interactive=False,
                    lines=10,
                    show_copy_button=True
                )

        # 添加示例提示
        gr.Examples(
            examples=[
                ["未来世界，机器人拥有了情感，他们"],
                ["在一个魔法王国里，会说话的猫"],
                ["如果时间旅行成为现实，历史学家会"]
            ],
            inputs=prompt_input,
            label="试试这些中文例子"
        )

        # 绑定按钮事件 - 只传递界面有的参数
        generate_btn.click(
            fn=generate_text,
            inputs=[
                prompt_input,
                max_length_slider,
                temperature_slider,
                top_p_slider,
                repetition_penalty_slider
            ],
            outputs=output_text
        )

    # 安全启动（自动端口检测）
    port = 7860
    while port < 8000:
        try:
            demo.launch(server_name="127.0.0.1", server_port=port)
            print(f"成功启动在端口: {port}")
            break
        except:
            print(f"端口 {port} 被占用，尝试下一个...")
            port += 1
    return

if __name__ == '__main__':
    main()