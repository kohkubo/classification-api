.PHONY: build
build:
	sam build --parallel --cached

.PHONY: deploy
deploy: build
	sam deploy --no-confirm-changeset

.PHONY: run
run: build
	sam local start-api -d 5858