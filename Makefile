.PHONY: clean run run-test run-cnn-test run-cnn-models train-cnn install test lint docs train-double-dqn train-double-dqn-advanced train-double-dqn-cnn train-double-dqn-cnn-advanced train-all-and-compare train-all-and-compare-advanced compare


# Installation
install:
	pip install -r requirements.txt

# Training commands
train-cnn:
	python train_dqn_cnn.py

train-dqn:
	python train_dqn_nn.py

train-double-dqn:
	python train_double_dqn.py

train-double-dqn-cnn:
	python train_double_dqn_cnn.py

train-double-dqn-cnn-advanced:
	@echo "Training Double DQN CNN with advanced parameters..."
	@read -p "Enter data path [data/your_data.csv]: " data_path; \
	data_path=$${data_path:-data/your_data.csv}; \
	read -p "Enter batch size [64]: " batch_size; \
	batch_size=$${batch_size:-64}; \
	read -p "Enter number of epochs [50]: " epochs; \
	epochs=$${epochs:-50}; \
	read -p "Enter learning rate [0.001]: " lr; \
	lr=$${lr:-0.001}; \
	read -p "Enter model name [double_dqn_cnn]: " model_name; \
	model_name=$${model_name:-double_dqn_cnn}; \
	DATA_PATH=$$data_path BATCH_SIZE=$$batch_size EPOCHS=$$epochs LEARNING_RATE=$$lr MODEL_NAME=$$model_name python train_double_dqn_cnn.py

train-double-dqn-advanced:
	@echo "Training Double DQN with advanced parameters..."
	@read -p "Enter data path [data/your_data.csv]: " data_path; \
	data_path=$${data_path:-data/your_data.csv}; \
	read -p "Enter batch size [64]: " batch_size; \
	batch_size=$${batch_size:-64}; \
	read -p "Enter number of epochs [50]: " epochs; \
	epochs=$${epochs:-50}; \
	read -p "Enter learning rate [0.001]: " lr; \
	lr=$${lr:-0.001}; \
	read -p "Enter model name [double_dqn]: " model_name; \
	model_name=$${model_name:-double_dqn}; \
	DATA_PATH=$$data_path BATCH_SIZE=$$batch_size EPOCHS=$$epochs LEARNING_RATE=$$lr MODEL_NAME=$$model_name python train_double_dqn.py

# Running models
run-cnn:
	python dqn_cnn.py

run-dqn:
	python dqn_nn.py

run-double-dqn:
	python double_dqn_nn.py

run-double-dqn-cnn:
	python double_dqn_cnn.py

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

# Train all models and generate comparison
train-all-and-compare: install
	@echo "Training all models and generating comparison..."
	@echo "Training DQN NN..."
	@python train_dqn_nn.py
	@echo "Training Double DQN NN..."
	@python train_double_dqn.py
	@echo "Training DQN CNN..."
	@python train_dqn_cnn.py
	@echo "Training Double DQN CNN..."
	@python train_double_dqn_cnn.py
	@echo "Generating comparison graphs..."
	@python compare_all_models.py
	@echo "All models trained and comparison generated!"

# Compare existing models or train if data is missing
compare: install
	@echo "Checking for existing model data in logs directory..."
	@if [ -d "logs/dqn_nn_logs" ] && [ -d "logs/double_dqn_nn_logs" ] && [ -d "logs/dqn_cnn_logs" ] && [ -d "logs/double_dqn_cnn_logs" ]; then \
		echo "All model data found. Generating comparison..." ; \
		python compare_all_models.py ; \
	else \
		echo "Some model data is missing. Running full training..." ; \
		$(MAKE) train-all-and-compare ; \
	fi

# Train all models with advanced parameters and generate comparison
train-all-and-compare-advanced: install
	@echo "Training all models with advanced parameters and generating comparison..."
	@echo "Training DQN NN with advanced parameters..."
	@read -p "Enter data path [data/your_data.csv]: " data_path; \
	data_path=$${data_path:-data/your_data.csv}; \
	read -p "Enter batch size [64]: " batch_size; \
	batch_size=$${batch_size:-64}; \
	read -p "Enter number of epochs [50]: " epochs; \
	epochs=$${epochs:-50}; \
	read -p "Enter learning rate [0.001]: " lr; \
	lr=$${lr:-0.001}; \
	read -p "Enter model name [dqn_nn]: " model_name; \
	model_name=$${model_name:-dqn_nn}; \
	DATA_PATH=$$data_path BATCH_SIZE=$$batch_size EPOCHS=$$epochs LEARNING_RATE=$$lr MODEL_NAME=$$model_name python train_dqn_nn.py
	@echo "Training Double DQN NN with advanced parameters..."
	@read -p "Enter data path [data/your_data.csv]: " data_path; \
	data_path=$${data_path:-data/your_data.csv}; \
	read -p "Enter batch size [64]: " batch_size; \
	batch_size=$${batch_size:-64}; \
	read -p "Enter number of epochs [50]: " epochs; \
	epochs=$${epochs:-50}; \
	read -p "Enter learning rate [0.001]: " lr; \
	lr=$${lr:-0.001}; \
	read -p "Enter model name [double_dqn_nn]: " model_name; \
	model_name=$${model_name:-double_dqn_nn}; \
	DATA_PATH=$$data_path BATCH_SIZE=$$batch_size EPOCHS=$$epochs LEARNING_RATE=$$lr MODEL_NAME=$$model_name python train_double_dqn.py
	@echo "Training DQN CNN with advanced parameters..."
	@read -p "Enter data path [data/your_data.csv]: " data_path; \
	data_path=$${data_path:-data/your_data.csv}; \
	read -p "Enter batch size [64]: " batch_size; \
	batch_size=$${batch_size:-64}; \
	read -p "Enter number of epochs [50]: " epochs; \
	epochs=$${epochs:-50}; \
	read -p "Enter learning rate [0.001]: " lr; \
	lr=$${lr:-0.001}; \
	read -p "Enter model name [dqn_cnn]: " model_name; \
	model_name=$${model_name:-dqn_cnn}; \
	DATA_PATH=$$data_path BATCH_SIZE=$$batch_size EPOCHS=$$epochs LEARNING_RATE=$$lr MODEL_NAME=$$model_name python train_dqn_cnn.py
	@echo "Training Double DQN CNN with advanced parameters..."
	@read -p "Enter data path [data/your_data.csv]: " data_path; \
	data_path=$${data_path:-data/your_data.csv}; \
	read -p "Enter batch size [64]: " batch_size; \
	batch_size=$${batch_size:-64}; \
	read -p "Enter number of epochs [50]: " epochs; \
	epochs=$${epochs:-50}; \
	read -p "Enter learning rate [0.001]: " lr; \
	lr=$${lr:-0.001}; \
	read -p "Enter model name [double_dqn_cnn]: " model_name; \
	model_name=$${model_name:-double_dqn_cnn}; \
	DATA_PATH=$$data_path BATCH_SIZE=$$batch_size EPOCHS=$$epochs LEARNING_RATE=$$lr MODEL_NAME=$$model_name python train_double_dqn_cnn.py
	@echo "Generating comparison graphs..."
	@python compare_all_models.py
	@echo "All models trained with advanced parameters and comparison generated!"