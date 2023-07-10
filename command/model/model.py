import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import LABELS, MODEL_PATH, PRETRAINED_MODEL_NAME


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(device, model_path=MODEL_PATH):
    """
    モデルを取得する関数
    初回train時にはローカルにモデルが存在しないため、huggingfaceからモデルを取得する
    """
    try:
        print("load model from local")
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH, num_labels=len(LABELS)
        ).to(device)
    except Exception:
        print("load model from huggingface")
        return AutoModelForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_NAME, num_labels=len(LABELS)
        ).to(device)


def get_tokenizer():
    return AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, use_fast=True)
