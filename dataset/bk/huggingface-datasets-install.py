from datasets import load_dataset, DatasetDict
import pandas as pd


def load_japanese_dataset() -> pd.DataFrame:
    # 日本語のデータセットを読み込む
    dataset_org: DatasetDict = load_dataset(
        "tyqiangz/multilingual-sentiments", "japanese"
    )
    dataset_org.set_format(type="pandas")
    return dataset_org


def save_csv_files(dataset: pd.DataFrame):
    # train, validation, testのデータをCSVファイルに保存する
    dataset["train"][:].to_csv("train.csv", index=None)
    dataset["validation"][:].to_csv("validation.csv", index=None)
    dataset["test"][:].to_csv("test.csv", index=None)


if __name__ == "__main__":
    dataset = load_japanese_dataset()
    save_csv_files(dataset)
    print("完了！")
