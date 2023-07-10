import argparse
from engine.train import train
from engine.eval import eval
from engine.predict import main as predict_main


def main():
    parser = argparse.ArgumentParser(
        description="Train, predict or evaluate the model."
    )
    subparsers = parser.add_subparsers(dest="mode")

    # Create a subparser for the "train" mode
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--train_file", type=str, required=True, help="Path to the training data file."
    )
    train_parser.add_argument(
        "--val_file", type=str, required=True, help="Path to the validation data file."
    )
    train_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory. ex) ../new_model",
    )

    # Create subparsers for the "predict" and "eval" modes (no additional arguments needed)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument(
        "--test_file", type=str, required=True, help="Path to the test data file."
    )
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument(
        "--test_file", type=str, required=True, help="Path to the test data file."
    )
    eval_parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="Path to the result data file.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(args.train_file, args.val_file, args.output_dir)
    elif args.mode == "predict":
        predict_main(args.test_file)
    elif args.mode == "eval":
        eval(args.test_file, args.result_file)
    else:
        print(
            f'Invalid mode {args.mode}. Please choose from "train", "predict", or "eval".'
        )


if __name__ == "__main__":
    main()
