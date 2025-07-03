# 安装核心库（最新版）
# !pip install transformers gradio torch
import os
import torch
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # 禁用符号链接警告
import gradio as gr
from transformers import pipeline
from opencc import OpenCC  # 用于简繁转换
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, AutoModelForCausalLM,AutoTokenizer
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import requests
from io import BytesIO

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

# 1. 创建生成器
## 文本生成模型
text_model_path = r"./models/gpt2-chinese-cluecorpussmall"
text_model = AutoModelForCausalLM.from_pretrained(text_model_path)
text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
text_generator = pipeline('text-generation', model=text_model,tokenizer=text_tokenizer, device=0)
print("加载成功!")

# 2. 加载多模态模型（BLIP图像描述生成）
blip_model_path = r"./models/blip-image-caption-base"
blip_model_path = r"./models/Taiyi_BLIP-Chinese"
blip_processor = BlipProcessor.from_pretrained(blip_model_path)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_path).to("cuda")
print("多模态模型加载成功!")



# 3. 多模态生成函数
def generate_multimodal(prompt, image, max_length=150, temperature=0.7, top_p=0.9, repetition_penalty=1.5,
                        mode="text_only"):
    try:
        # 模式选择
        if mode == "text_only" or image is None:
            # 纯文本生成
            params = {
                "max_length": int(max_length),
                "num_return_sequences": 1,
                "temperature": float(temperature),
                "top_k": 50,
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "do_sample": True,
            }
            result = text_generator(prompt, **params)[0]['generated_text']
            ##简中转繁中
            simplified_result = cc.convert(result)
            return simplified_result

        elif mode == "image_caption":
            # 图像描述生成
            inputs = blip_processor(image, prompt, return_tensors="pt").to("cuda")
            outputs = blip_model.generate(**inputs, max_new_tokens=int(max_length))
            caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
            ##简中转繁中
            simplified_result = cc.convert(caption)
            return simplified_result

        elif mode == "multimodal_story":
            # 多模态故事生成：先获取图像描述，再基于描述生成故事
            # 第一步：生成图像描述
            inputs = blip_processor(image, "", return_tensors="pt").to("cuda")
            outputs = blip_model.generate(**inputs, max_new_tokens=50)
            image_description = blip_processor.decode(outputs[0], skip_special_tokens=True)

            # 第二步：基于描述和提示生成故事
            combined_prompt = f"图片描述: {image_description}\n用户提示: {prompt}\n故事:"

            story_params = {
                "max_length": int(max_length),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "do_sample": True,
            }
            story = text_generator(combined_prompt, **story_params)[0]['generated_text']
            ##简中转繁中
            simplified_result = cc.convert(f"图像描述: {image_description}\n\n生成故事:\n{story}")
            return simplified_result
            # 返回完整结果（包含描述和故事）
            # return cc.convert(f"图像描述: {image_description}\n\n生成故事:\n{story}")

    except Exception as e:
        return f"生成失败: {str(e)}"

# 4. 下载示例图片的函数
def download_image(url):
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    except:
        return None

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
        result = text_generator(prompt, **params)[0]['generated_text']
        # 简繁转换（确保输出简体中文）
        simplified_result = cc.convert(result)
        return simplified_result
    except Exception as e:
        return f"生成失败: {str(e)}"


def main():
    # 示例图片URL
    example_images = [
        r"resource/Bao-0000.jpg",
        r"resource/Bao-0001.jpg",
        r"resource/Bao-0002.jpg",
        r"resource/Bao-0003.jpg",
        r"resource/Bao-0004.jpg",
        r"resource/Bao-0005.jpg",
        r"resource/Bao-0006.jpg",
        r"resource/Bao-0007.jpg",
        r"resource/Bao-0008.jpg",
        r"resource/Bao-0009.jpg",
    ]

    with gr.Blocks(title="多模态AI创作实验室") as demo:
        gr.Markdown("## 🎨 多模态创作工作台 - 文本与图像联合生成")

        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=1):
                mode_radio = gr.Radio(
                    choices=["text_only", "image_caption", "multimodal_story"],
                    value="text_only",
                    label="创作模式",
                    info="选择生成模式"
                )

                prompt_input = gr.Textbox(
                    label="输入提示",
                    placeholder="在这里输入你的创意开头...",
                    lines=3
                )

                image_input = gr.Image(
                    type="pil",
                    label="上传图片",
                    interactive=True
                )

                with gr.Accordion("高级参数", open=False):
                    max_length_slider = gr.Slider(
                        minimum=50, maximum=300, value=150, step=10,
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

                generate_btn = gr.Button("开始创作", variant="primary")

            # 右侧：输出区
            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="创作结果",
                    interactive=False,
                    lines=12,
                    show_copy_button=True
                )
                gr.Markdown("### 模式说明:")
                gr.Markdown("- **纯文本模式**: 仅基于文本提示生成内容")
                gr.Markdown("- **图像描述模式**: 根据图片生成描述文字")
                gr.Markdown("- **多模态故事模式**: 结合图片和提示生成创意故事")

        # 示例区
        with gr.Row():
            gr.Examples(
                examples=[
                    ["未来世界，机器人拥有了情感，他们", None, "text_only"],
                    ["描述这张图片中的场景", example_images[0], "image_caption"],
                    ["根据图片编一个科幻故事", example_images[2], "multimodal_story"]
                ],
                inputs=[prompt_input, image_input, mode_radio],
                label="尝试这些示例",
                examples_per_page=3
            )

        # 绑定按钮事件
        generate_btn.click(
            fn=generate_multimodal,
            inputs=[
                prompt_input,
                image_input,
                max_length_slider,
                temperature_slider,
                top_p_slider,
                repetition_penalty_slider,
                mode_radio
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