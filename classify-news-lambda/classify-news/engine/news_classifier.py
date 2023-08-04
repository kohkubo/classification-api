import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import yaml
import os


class NewsClassifier:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
            cfg = yaml.safe_load(f)

        self.labels = cfg["labels"]
        self.num_labels = len(self.labels)
        self.model_path = cfg["model_path"]
        self.pretrained_model_name = cfg["pretrained_model_name"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=self.num_labels
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name, use_fast=True
        )

    def classify_news(self, sentence):
        """
        文章を入力すると、ラベルと予測確率を返す関数

        Args:
            sentence (str): 予測したい文章

        Returns:
            tuple: ラベル番号、ラベル名、予測確率の配列、最大予測確率
        """
        # 文章をトークン化し、PyTorchのテンソルに変換する
        inputs = self.tokenizer(sentence, return_tensors="pt")

        # テンソルをモデルに入力し、予測を行う
        with torch.no_grad():
            outputs = self.model(
                inputs["input_ids"].to(self.device),
                inputs["attention_mask"].to(self.device),
            )

            # 予測結果をソフトマックス関数で正規化する
            prediction = torch.nn.functional.softmax(outputs.logits, dim=1)

        # 最も確率の高いラベル番号を計算する
        argmax_prediction = torch.argmax(prediction)

        # 予測確率の配列をCPUに移動し、numpy配列に変換する(npはgpuに対応していないため)
        prediction_np = prediction.cpu().detach().numpy()
        predict = [
            f"{self.labels[i]}: {prediction_np[0][i]}" for i in range(self.num_labels)
        ]

        # ラベル番号、ラベル名、予測確率の配列、最大予測確率を返す
        return (
            int(argmax_prediction),
            self.labels[int(argmax_prediction)],
            predict,
            np.max(prediction_np),
        )
