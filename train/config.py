import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = [
    "dokujo",
    "it",
    "sports",
    "kaden",
    "homme",
    "movie",
    "peachy",
    "smax",
    "topic-news",
]

MODEL_PATH = "../news_model"
TEST_PATH = "../text/news_test.csv"
TEST_RESULT_PATH = "../text/news_test_result.csv"
PRETRAINED_MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"


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
