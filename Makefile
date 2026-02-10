.PHONY: install format lint typecheck test build deploy check clean

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	$(PIP) install -e ".[dev]"

format:
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

typecheck:
	$(VENV)/bin/mypy src/ bench.py

test:
	$(PYTHON) -m pytest tests/ -v

build:
	docker build -t openclaw-autorouter .

deploy:
	./build-and-push.sh

check: lint typecheck test

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
