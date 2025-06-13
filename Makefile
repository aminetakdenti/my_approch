.PHONY: clean run run-test run-cnn-test run-cnn-models train-cnn install test lint docs


# Installation
install:
	pip install -r requirements.txt

# Training commands
train-cnn:
	python train_dqn_cnn.py

train-dqn:
	python train_dqn_nn.py

# Running models
run-cnn:
	python dqn_cnn.py

run-dqn:
	python dqn_nn.py

# Testing
test:
	pytest tests/

# Linting
lint:
	flake8 src/
	black src/
	isort src/

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