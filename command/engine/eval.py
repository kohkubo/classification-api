from sklearn.metrics import classification_report

import pandas as pd

from config import LABELS


def eval(test_path, test_result_path):
    test_df = pd.read_csv(test_path, encoding="UTF-8")
    test_result_df = pd.read_csv(test_result_path, encoding="UTF-8")

    # テストデータの正解ラベルと予測ラベルを用いて混同行列をプロットする
    y_true = []
    y_pred = []
    for i in range(len(test_df)):
        y_true.append(test_df["label"][i])
        y_pred.append(test_result_df["label"][i])

    # クラス分類の評価指標を計算し、DataFrameに保存する
    report = classification_report(
        y_true, y_pred, target_names=LABELS, output_dict=True
    )
    report_df = pd.DataFrame(report).T
    print(report_df)
