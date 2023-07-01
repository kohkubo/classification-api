from transformers import TrainingArguments
from transformers import Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from datasets import DatasetDict
from sklearn.metrics import classification_report
from config import (
    LABELS,
    MODEL_PATH,
)
from model import get_device, get_model, get_tokenizer


def train():
    device = get_device()
    model = get_model(device)
    tokenizer = get_tokenizer()

    # %%
    # 訓練データと検証データを読み込む
    train_df = pd.read_csv("text/news_train.csv", encoding="UTF-8")
    validation_df = pd.read_csv("text/news_val.csv", encoding="UTF-8")

    # pandas DataFrameからDatasetオブジェクトを作成する
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(validation_df)

    # DatasetDictオブジェクトを作成する
    dataset = DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
        }
    )

    # %%
    # トークナイザ処理
    def tokenize(batch):
        return tokenizer(batch["sentence"], padding=True, truncation=True)

    # Datasetオブジェクトをトークナイズする
    dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)

    # %%
    # 評価指標の定義
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    # %%
    # 学習の準備
    batch_size = 16
    logging_steps = len(dataset_encoded["train"]) // batch_size
    model_name = "classification"
    training_args = TrainingArguments(
        output_dir=model_name,
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=False,
        log_level="error",
    )

    # %%
    # 学習
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    # %%
    # 検証データで予測を行う
    preds_output = trainer.predict(dataset_encoded["validation"])

    # 予測結果を取得する
    y_preds = np.argmax(preds_output.predictions, axis=1)
    y_valid = np.array(dataset_encoded["validation"]["label"])

    # %%
    # 評価指標を表示する
    print(f"Accuracy: {accuracy_score(y_valid, y_preds)}")
    print(f"F1: {f1_score(y_valid, y_preds, average='weighted')}")
    print(f"Confusion Matrix: \n{confusion_matrix(y_valid, y_preds)}")
    print(f"Classification Report: \n{classification_report(y_valid, y_preds)}")
    print(f"Labels: {LABELS}")
    print(f"Predictions: {y_preds}")
    print(f"True Values: {y_valid}")

    # %%
    # モデルを保存する
    trainer.save_model(MODEL_PATH)
