import yaml

CONFIG = None

# YAMLファイルから設定を読み込む
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

LABELS = CONFIG["labels"]
MODEL_PATH = CONFIG["model_path"]
TEST_RESULT_PATH = CONFIG["test_result_path"]
PRETRAINED_MODEL_NAME = CONFIG["pretrained_model_name"]
