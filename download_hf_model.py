# 将模型下载到本地调用
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

# 将模型和分词器下载到本地，并指定保存路径
model_name = "IDEA-CCNL/Taiyi-BLIP-750M-Chinese"#模型的名字
cache_dir = "./models"

# 下载模型
AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
# 下载分词工具
AutoModelForImageTextToText.from_pretrained(model_name, cache_dir=cache_dir)

print(f"模型分词器已下载到：{cache_dir}")