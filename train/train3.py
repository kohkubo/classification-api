import torch
import pandas as pd
import numpy as np
from lime.lime_text import LimeTextExplainer
from config import g_model, g_tokenizer, g_device


# テストデータの取得
test_df = pd.read_csv("./text/news_test.csv", encoding="UTF-8")


def predictor(sample_text):
    # 与えられたテキストをトークン化し、前処理を行う
    inputs = g_tokenizer(
        sample_text,
        return_tensors="pt",
        add_special_tokens=True,
        padding="max_length",
        max_length=512,
        truncation=True,
    )
    # PyTorchモデルを評価モードに設定する
    g_model.eval()
    # 勾配計算を行わないようにする
    with torch.no_grad():
        # モデルに入力を与えて予測を行う
        outputs = g_model(
            inputs["input_ids"].to(g_device), inputs["attention_mask"].to(g_device)
        )
        # 予測された確率を計算し、NumPy配列として返す
        probas = (
            torch.nn.functional.softmax(outputs.logits, dim=1).cpu().detach().numpy()
        )
        return probas


# ニュース分類
class_names = [
    "dokujo-tsushin",
    "it-life-hack",
    "smax",
    "sports-watch",
    "kaden-channel",
    "movie-enter",
    "topic-news",
    "livedoor-homme",
    "peachy",
]


# テストデータセットからテキストと正解ラベルを取得
text = test_df["sentence"][14]
label = test_df["label"][14]

# LIME入力用
texts = []
texts.append(text)

output = predictor(texts)
print(output)
print("予測", class_names[np.argmax(output)])
print("正解", class_names[label])


explainer = LimeTextExplainer(class_names=class_names)

# 予測確率が高いTOP-K
exp = explainer.explain_instance(
    text, predictor, num_features=10, num_samples=70, top_labels=5
)
exp.show_in_notebook(text=text)

# 結果を保存する
exp.save_to_file("explain.html")
