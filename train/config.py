import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yaml

# YAMLファイルから設定を読み込む
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

LABELS = config["labels"]
MODEL_PATH = config["model_path"]
TEST_PATH = config["test_path"]
TEST_RESULT_PATH = config["test_result_path"]
PRETRAINED_MODEL_NAME = config["pretrained_model_name"]


def get_model(device):
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


# g_ はグローバル変数
g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g_model = get_model(g_device)
g_tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, use_fast=True)
