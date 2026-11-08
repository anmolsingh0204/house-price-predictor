# House Price Predictor - Makefile
# Common development and deployment commands

.PHONY: help install test lint format train run-api run-ws run-dev docker-build docker-run clean

# Default target
help:
	@echo "House Price Predictor - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo "  make install-dev      Install with dev dependencies"
	@echo "  make setup            Full setup (install + generate data + train)"
	@echo ""
	@echo "Data & Training:"
	@echo "  make generate-data    Generate synthetic dataset"
	@echo "  make train            Train ML model"
	@echo "  make train-quick      Quick training (1000 samples)"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test             Run all tests"
	@echo "  make test-cov         Run tests with coverage"
	@echo "  make lint             Run linters (flake8, mypy)"
	@echo "  make format           Format code with black"
	@echo "  make check            Run all quality checks"
	@echo ""
	@echo "Running Services:"
	@echo "  make run-api          Start REST API server"
	@echo "  make run-ws           Start WebSocket server"
	@echo "  make run-dev          Start both with auto-reload"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run with docker-compose"
	@echo "  make docker-dev       Run development environment"
	@echo "  make docker-test      Run tests in Docker"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Clean generated files"
	@echo "  make clean-all        Clean everything including models"

# Variables
PYTHON = python
PIP = pip
PYTEST = pytest
BLACK = black
FLAKE8 = flake8
MYPY = mypy

# Setup
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov pytest-mock pytest-asyncio black flake8 mypy pre-commit pip-audit

setup: install generate-data train
	@echo "Setup complete!"

# Data & Training
generate-data:
	$(PYTHON) src/data/generate_dataset.py

train:
	$(PYTHON) src/ml/train_model.py

train-quick:
	$(PYTHON) -c "
import sys
sys.path.insert(0, 'src/ml')
from train_model import generate_dataset
import pandas as pd
df = generate_dataset(1000)
df.to_csv('data/train_data.csv', index=False)
df.to_csv('data/test_data.csv', index=False)
print('Quick train data generated')
"
	$(PYTHON) src/ml/train_model.py

# Testing
test:
	$(PYTEST) src/tests/ -v

test-cov:
	$(PYTEST) src/tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	$(PYTEST) src/tests/ -v -m "not integration"

test-integration:
	$(PYTEST) src/tests/ -v -m integration

# Quality Checks
lint:
	$(FLAKE8) src/ --max-line-length=100
	$(MYPY) src/ --ignore-missing-imports || true

format:
	$(BLACK) src/ --line-length=100

check: format lint test
	@echo "All checks passed!"

# Running Services
run-api:
	$(PYTHON) src/api/app.py

run-ws:
	$(PYTHON) src/websocket/ws_server.py

run-dev:
	@echo "Starting API and WebSocket in background..."
	$(PYTHON) src/api/app.py & $(PYTHON) src/websocket/ws_server.py

# Docker
docker-build:
	docker build -t house-price-predictor:latest .

docker-run:
	docker-compose up -d

docker-dev:
	docker-compose --profile dev up dev

docker-test:
	docker-compose --profile test up test

docker-train:
	docker-compose --profile training up ml-trainer

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf data/*.csv models/*.pkl models/*.json venv .venv dist build *.egg-info
	@echo "All generated files cleaned!"

# Pre-commit
pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

# Release
version-patch:
	@echo "Bump patch version (manual)"

version-minor:
	@echo "Bump minor version (manual)"

version-major:
	@echo "Bump major version (manual)"