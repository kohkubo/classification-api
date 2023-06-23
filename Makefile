DATATIME=$(shell date "+%Y%m%d%H%M%S")

.PHONY: livedoor-news-corpus-init
livedoor-news-corpus-init	:
	@echo "livedoorニュースコーパスのダウンロード"
	wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
	tar zxvf ldcc-20140209.tar.gz
	rm ldcc-20140209.tar.gz

.PHONY: huggingface-datasets
huggingface-datasets	:
	@echo "huggingface/datasetsのインストール"
	python huggingface-datasets-install.py

.PHONY: run
run	:
	@echo "fastapi起動"
	uvicorn src.main:app --reload

.PHONY: train
train	:
	@echo "学習"
	cd train && python train.py > logs/train_$(DATATIME).log

.PHONY: predict
predict	:
	@echo "予測"
	cd train && python predict.py > logs/predict_$(DATATIME).log

.PHONY: eval
eval	:
	@echo "評価"
	cd train && python eval.py