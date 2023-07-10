.PHONY: all
all	: train predict eval

.PHONY: train
train	:
	@echo "学習"
	python main.py train \
		--train_file=../text/news_train.csv \
		--val_file=../text/news_val.csv \
		--output_dir=../news_model

.PHONY: predict
predict	:
	@echo "予測"
	python main.py predict \
		--test_file=../text/news_test.csv

.PHONY: eval
eval	:
	@echo "予測"
	python main.py eval \
		--test_file=../text/news_test.csv \
		--result_file=../text/news_test_result.csv
