# verify_model.py

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

# 设置模型路径
MODEL_PATH = './deberta-v3-base-lora-finetuned'

# 加载分词器
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("分词器加载成功")
except Exception as e:
    print("分词器加载失败:", e)

# 加载基础模型
try:
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print("基础模型加载成功")
except Exception as e:
    print("基础模型加载失败:", e)

# 加载LoRA微调模型
try:
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    print("LoRA模型加载成功")
except Exception as e:
    print("LoRA模型加载失败:", e)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 创建预测pipeline
from transformers import pipeline

try:
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        function_to_apply='sigmoid',
        device=0 if torch.cuda.is_available() else -1
    )
    print("预测pipeline创建成功")
except Exception as e:
    print("预测pipeline创建失败:", e)

# 测试预测
try:
    test_text = "This is a test comment to classify."
    results = classifier(test_text)
    print("预测结果:", results)
except Exception as e:
    print("预测失败:", e)
