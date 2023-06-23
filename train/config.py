import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = [
    "dokujo",
    "it",
    "sports-",
    "kaden",
    "homme",
    "movie",
    "peachy",
    "smax",
    "topic-news",
]

MODEL_PATH = "../text/news_model"

TEST_PATH = "../text/news_test.csv"

TEST_RESULT_PATH = "../text/news_test_result.csv"

tokenizer = AutoTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)


g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=len(LABELS)
).to(g_device)
g_tokenizer = AutoTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
