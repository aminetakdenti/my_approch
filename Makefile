.PHONY: setup create-dataset run-hybrid-dqn run-all clean run run-test run-cnn-test run-cnn-models train-cnn install test lint docs

# Setup
setup:
	mkdir -p src/utils src/logs data/raw data/processed data/external models/cnn models/dqn models/hybrid logs/cnn logs/dqn logs/hybrid

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

# Dataset creation
create-dataset: setup
	python src/data/preprocessing.py

# Training commands
train-cnn: setup
	python src/training/train_cnn.py

train-dqn: setup
	python src/training/train_dqn.py

train-hybrid: setup
	python src/training/train_hybrid.py

# Running models
run-cnn: setup
	python src/models/cnn.py

run-dqn: setup
	python src/models/dqn.py

run-hybrid: setup
	python src/models/hybrid.py

# Testing
test:
	pytest tests/

# Linting
lint:
	flake8 src/
	black src/
	isort src/

# Documentation
docs:
	cd docs && make html

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf models/*.pth
	rm -rf logs/*.log
	rm -rf data/processed/*
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

# Development
dev: install lint test

# All
all: setup install test lint docs