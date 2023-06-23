import torch
import numpy as np

from engine.config import (
    LABELS,
    g_model,
    g_tokenizer,
    g_device,
)


def classify_news(sentence):
    """
    文章を入力すると、ラベルと予測確率を返す関数

    Args:
        sentence (str): 予測したい文章

    Returns:
        tuple: ラベル番号、ラベル名、予測確率の配列、最大予測確率
    """
    # 文章をトークン化し、PyTorchのテンソルに変換する
    inputs = g_tokenizer(sentence, return_tensors="pt")

    # テンソルをモデルに入力し、予測を行う
    with torch.no_grad():
        outputs = g_model(
            inputs["input_ids"].to(g_device), inputs["attention_mask"].to(g_device)
        )

        # 予測結果をソフトマックス関数で正規化する
        prediction = torch.nn.functional.softmax(outputs.logits, dim=1)

    # 最も確率の高いラベル番号を計算する
    argmax_prediction = torch.argmax(prediction)

    # 予測確率の配列をCPUに移動し、numpy配列に変換する(npはgpuに対応していないため)
    prediction_np = prediction.cpu().detach().numpy()
    predict = [f"{LABELS[i]}: {prediction_np[0][i]}" for i in range(len(LABELS))]

    # ラベル番号、ラベル名、予測確率の配列、最大予測確率を返す
    return (
        int(argmax_prediction),
        LABELS[int(argmax_prediction)],
        predict,
        np.max(prediction_np),
    )
