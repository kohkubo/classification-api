import torch
import pandas as pd
import numpy as np
from config import (
    LABELS,
    TEST_PATH,
    TEST_RESULT_PATH,
)
from model import get_model, get_tokenizer, get_device

device = get_device()
model = get_model(device)
tokenizer = get_tokenizer()

# テストデータを読み込む
test_df = pd.read_csv(TEST_PATH, encoding="UTF-8")


def predict_label(sentence):
    """
    文章を入力すると、ラベルと予測確率を返す関数

    Args:
        sentence (str): 予測したい文章

    Returns:
        tuple: ラベル番号、ラベル名、予測確率の配列、最大予測確率
    """
    # 文章をトークン化し、PyTorchのテンソルに変換する
    inputs = tokenizer(sentence, return_tensors="pt")

    # テンソルをモデルに入力し、予測を行う
    with torch.no_grad():
        outputs = model(
            inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
        )

        # 予測結果をソフトマックス関数で正規化する
        prediction = torch.nn.functional.softmax(outputs.logits, dim=1)

    # 最も確率の高いラベル番号を計算する
    argmax_prediction = torch.argmax(prediction)

    # 予測確率の配列をCPUに移動し、numpy配列に変換する(npはgpuに対応していないため)
    prediction_np = prediction.cpu().detach().numpy()

    # ラベル番号、ラベル名、予測確率の配列、最大予測確率を返す
    return (
        int(argmax_prediction),
        LABELS[int(argmax_prediction)],
        prediction_np,
        np.max(prediction_np),
    )


# テストデータの各文章に対して予測を行い、結果をDataFrameに保存する
# この処理はバッチ処理で高速化可能だが、今回はfor文で処理している
df = pd.DataFrame(columns=["label", "label_name", "pred", "sentence"])
for sentence in test_df["sentence"]:
    label, label_name, prediction, max_prob = predict_label(sentence)
    new_row = pd.DataFrame(
        {
            "label": [label],
            "label_name": [label_name],
            "pred": [max_prob],
            "sentence": [sentence],
        }
    )

    df = pd.concat([df, new_row], ignore_index=True)

# 予測結果を保存する
print(f"save result to {TEST_RESULT_PATH}")
df.to_csv(TEST_RESULT_PATH, encoding="UTF-8", index=False)
