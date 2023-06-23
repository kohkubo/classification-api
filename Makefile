DATATIME=$(shell date "+%Y%m%d%H%M%S")

.PHONY: livedoor-news-corpus-init
livedoor-news-corpus-init	:
	@echo "livedoorニュースコーパスのダウンロード"
	wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
	tar zxvf ldcc-20140209.tar.gz
	rm ldcc-20140209.tar.gz

.PHONY: setup
setup	: train
	@echo "環境構築"
	@echo "app/engineにモデルを配置"
	cp -r news_model/ app/engine/

.PHONY: run
run	:
	@echo "fastapi起動"
	cd app && uvicorn main:app --reload

.PHONY: train
train	:
	@echo "学習"
	cd train && python train.py > ../logs/train_$(DATATIME).log

.PHONY: predict
predict	: check-model-exit
	@echo "予測"
	cd train && python predict.py > ../logs/predict_$(DATATIME).log

.PHONY: eval
eval	: check-model-exit
	@echo "評価"
	cd train && python eval.py

.PHONY: check-model-exit
check-model-exit	:
	@echo "モデルの存在確認"
	@if [ ! -d news_model ]; then \
		echo "モデルが存在しません。make trainで学習を実行してください。"; \
		exit 1; \
	fi

.PHONY: api-test
api-test	:
	@echo "APIテスト"
	curl -X POST -H "Content-Type: application/json" -d '{"title":"女性を潤す新たな注目ワードは“アミノ酸”"}' http://localhost:8000/v1/classify_news_type/
