import argparse
from train import train
from predict import predict_dataset
from eval import eval


def main():
    parser = argparse.ArgumentParser(
        description="Train, predict or evaluate the model."
    )
    parser.add_argument(
        "mode",
        type=str,
        help='Mode to run the script in: "train", "predict", or "eval"',
    )

    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "predict":
        predict_dataset()
    elif args.mode == "eval":
        eval()
    else:
        print(
            f'Invalid mode {args.mode}. Please choose from "train", "predict", or "eval".'
        )


if __name__ == "__main__":
    main()
