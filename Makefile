.PHONY: install test lint ci index query doctor eval docker-build docker-up docker-down

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

index:
	kairo index

query:
	kairo query "请概述 KairoRAG 的能力。"

doctor:
	kairo doctor

eval:
	kairo eval --suite all --fake-providers

docker-build:
	docker build -t kairorag .

docker-up:
	docker compose up --build

docker-down:
	docker compose down
