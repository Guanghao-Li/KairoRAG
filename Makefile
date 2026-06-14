.PHONY: install test lint ci index query doctor doctor-release eval eval-compare ui live-test docker-build docker-up docker-down docker-ui docker-eval

install:
	python -m pip install -e ".[dev,web]"

test:
	python -m pytest

lint:
	python -m ruff check .

ci: lint test
	python -m kairorag.cli --help
	python -m kairorag.cli doctor --help
	python -m kairorag.cli eval --help
	python -m kairorag.cli ui --help

index:
	kairo index

query:
	kairo query "请概述 KairoRAG 的能力。"

doctor:
	kairo doctor

doctor-release:
	kairo doctor --release

eval:
	kairo eval --suite all --fake-providers

eval-compare:
	kairo eval --compare baseline.json current.json --output-dir data/eval_compare

ui:
	kairo ui --host 127.0.0.1 --port 8000

live-test:
	RUN_LIVE_PROVIDER_TESTS=1 python -m pytest tests/live -m live

docker-build:
	docker build -t kairorag .

docker-up:
	docker compose --profile app up --build

docker-ui:
	docker compose --profile app up --build ui

docker-eval:
	docker compose --profile eval run --rm eval

docker-down:
	docker compose down
