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
