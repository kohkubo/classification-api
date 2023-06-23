import os
from typing import List
import pandas as pd


def get_title_list(path: str) -> List[str]:
    """
    指定されたパスのファイルからタイトルのリストを取得する。
    """
    title_list: List[str] = []
    filenames: List[str] = os.listdir(path)
    for filename in filenames:
        with open(path + filename) as f:
            # 元データ参照 3行目がタイトル
            title: str = f.readlines()[2].strip()
            title_list.append(title)
    return title_list


def generate_dataset() -> pd.DataFrame:
    """
    テキストファイル内のタイトルからデータセットを生成する。
    """
    df: pd.DataFrame = pd.DataFrame(columns=["label", "sentence"])

    # カテゴリーのリスト
    categories = [
        "dokujo-tsushin",
        "it-life-hack",
        "sports-watch",
        "kaden-channel",
        "livedoor-homme",
        "movie-enter",
        "peachy",
        "smax",
        "topic-news",
    ]

    DIR_PATH = "../text/"

    for i, category in enumerate(categories):
        title_list = get_title_list(DIR_PATH + category + "/")
        df = pd.concat([df, pd.DataFrame({"label": i, "sentence": title_list})])

    df = df.sample(frac=1)  # データをシャッフルする

    num = len(df)
    # スライスを使用してデータ分割、train: 70%, val: 20%, test: 10%
    df[: int(num * 0.7)].to_csv(
        DIR_PATH + "news_train.csv", sep=",", index=False
    )  # 70%をトレーニングデータとして保存する
    df[int(num * 0.7) : int(num * 0.9)].to_csv(
        DIR_PATH + "news_val.csv", sep=",", index=False
    )  # 20%をバリデーションデータとして保存する
    df[int(num * 0.9) :].to_csv(
        DIR_PATH + "news_test.csv", sep=",", index=False
    )  # 10%をテストデータとして保存する

    return df


if __name__ == "__main__":
    df = generate_dataset()
