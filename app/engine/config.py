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

MODEL_PATH = "./engine/news_model"

PRETRAINED_MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"

# g_ はグローバル変数
g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=len(LABELS)
).to(g_device)
g_tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, use_fast=True)
