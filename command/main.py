import argparse
from train import train
from predict import predict_dataset
from eval import eval


def main():
    parser = argparse.ArgumentParser(
        description="Train, predict or evaluate the model."
    )
    subparsers = parser.add_subparsers(dest="mode")

    # Create a subparser for the "train" mode
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to the training data file."
    )
    train_parser.add_argument(
        "--val_file",
        type=str,
        required=True,
        help="Path to the validation data file."
    )

    # Create subparsers for the "predict" and "eval" modes (no additional arguments needed)
    subparsers.add_parser("predict")
    subparsers.add_parser("eval")

    args = parser.parse_args()

    if args.mode == "train":
        train(args.train_file, args.val_file)
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
