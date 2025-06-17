# Deep Learning Models Implementation

This repository contains implementations of Convolutional Neural Networks (CNN) and Deep Q-Networks (DQN) for various tasks.

## Project Structure

```
.
├── cnn_models.py      # CNN model implementation with visualization
├── dqn_cnn.py        # DQN with CNN architecture
├── dqn_pytorch.py    # PyTorch DQN implementation
├── hybrid_dqn_agent.py# Hybrid DQN implementation
├── requirements.txt   # Project dependencies
└── data/            # Data directory
```

## Setup and Installation

1. Create a virtual environment and activate it:
```bash
python -m venv env
source env/bin/activate  # On Linux/Mac
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## CNN Model Usage

The CNN model (`cnn_models.py`) provides a flexible implementation with built-in visualization tools.

### Basic Usage

```python
from cnn_models import CNNModel, CNNTrainer

# Create model
model = CNNModel(
    input_channels=1,  # Number of input channels (1 for grayscale, 3 for RGB)
    height=28,         # Input height
    width=28,         # Input width
    output_dim=10     # Number of classes
)

# Create trainer
trainer = CNNTrainer(model, learning_rate=0.001)

# Train the model
losses, accuracies, best_acc = trainer.run_training(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    save_dir='model_checkpoints',
    early_stopping_patience=3
)
```

### Visualization Features

The model includes several visualization tools:

1. Training Metrics:
   - Loss and accuracy plots over time
   - Moving averages for smoother visualization

2. Confusion Matrix:
   - Visual representation of model predictions
   - Automatically generated during validation

3. Per-Class Metrics:
   - Precision, recall, and F1-score for each class
   - Bar plots for easy comparison

### Example with MNIST

```python
import torch.utils.data as data
from torchvision import datasets, transforms

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Create data loaders
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=32)

# Train model
model = CNNModel(input_channels=1, height=28, width=28, output_dim=10)
trainer = CNNTrainer(model)
trainer.run_training(
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=10,
    save_dir='mnist_training'
)
```

## DQN Model Usage

The repository includes several DQN implementations:

1. Basic DQN (`dqn_pytorch.py`)
2. CNN-based DQN (`dqn_cnn.py`)
3. Hybrid DQN (`hybrid_dqn_agent.py`)

### Basic DQN Usage

```python
from dqn_pytorch import DeepQNet, DataLoader

# Load data
dataset = DataLoader("data/dataset.csv", batch_size=64)

# Create and train DQN
dqn = DeepQNet(dataset)
dqn.train(epochs=100)

# Evaluate
accuracy, precision, recall, f1 = dqn.test()
```

### Hybrid DQN Usage

```python
from hybrid_dqn_agent import HybridDQN, DataLoader

# Initialize dataset
dataset = DataLoader(
    "data/dataset.csv",
    batch_size=64,
    img_channels=1
)

# Create and train model
dqn = HybridDQN(
    dataset,
    img_size=7,  # Input image size
    img_channels=1
)

# Train model
losses, accuracies = dqn.train()

# Evaluate
accuracy, precision, recall, f1 = dqn.test()
```

## Model Visualization

All models include comprehensive visualization tools in the `mnist_training/plots` directory:

- Training metrics (loss and accuracy)
- Confusion matrices
- Per-class performance metrics
- Class distribution plots

## Advanced Features

1. Early Stopping:
   - Prevents overfitting
   - Configurable patience parameter

2. Learning Rate Scheduling:
   - Reduces learning rate on plateau
   - Improves convergence

3. Class Weight Balancing:
   - Handles imbalanced datasets
   - Automatically computed from data

4. Batch Normalization:
   - Improves training stability
   - Faster convergence

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.

## License

This project is open-source and available under the MIT License.

# Double DQN CNN Implementation

This project implements a Double Deep Q-Network (Double DQN) using Convolutional Neural Networks (CNN) for reinforcement learning tasks. The implementation combines the benefits of Double DQN with the feature extraction capabilities of CNNs.

## Architecture Overview

### 1. DoubleCNNModel Class
The `DoubleCNNModel` class implements the neural network architecture with two key components:

- **Main Network**: The primary CNN that learns the Q-values
- **Target Network**: A separate CNN that provides stable Q-value estimates

The model uses a 2D CNN architecture because:
- It processes input data as 2D images (reshaped from feature vectors)
- Each feature vector is reshaped into a square matrix
- The CNN can learn spatial patterns in the data

### 2. DoubleCNNTrainer Class
The `DoubleCNNTrainer` class handles the training process and implements the Double DQN algorithm. It:
- Manages the training loop
- Updates both networks
- Handles data preprocessing
- Implements early stopping and learning rate scheduling

## How Double DQN Works

1. **Action Selection**:
   - The main network selects actions based on current state
   - Uses the Q-values from the main network to choose actions

2. **Value Estimation**:
   - The target network estimates the Q-values for the next state
   - This separation reduces overestimation of Q-values

3. **Network Updates**:
   - Main network is updated every step
   - Target network is updated slowly using soft updates
   - Soft updates help maintain stability in training

## Key Features

### Model Architecture
```python
# Main Network
- Input Layer (2D)
- Convolutional Layers (2D)
- Batch Normalization
- Max Pooling
- Dropout
- Fully Connected Layers

# Target Network (Mirror of Main Network)
- Same architecture as main network
- Updated slowly to maintain stability
```

### Training Process
1. **Data Preparation**:
   - Input features are reshaped into 2D images
   - Data is normalized and batched

2. **Training Loop**:
   - Forward pass through main network
   - Action selection using main network
   - Q-value estimation using target network
   - Loss calculation and backpropagation
   - Soft update of target network

3. **Monitoring**:
   - Training metrics tracking
   - Validation performance
   - Early stopping
   - Learning rate scheduling

## Usage

### Basic Training
```bash
make train-double-dqn-cnn
```

### Advanced Training with Custom Parameters
```bash
make train-double-dqn-cnn-advanced
```

### Running the Model
```bash
make run-double-dqn-cnn
```

## Why Two Classes?

The separation into `DoubleCNNModel` and `DoubleCNNTrainer` follows the Single Responsibility Principle:

1. **DoubleCNNModel**:
   - Defines the neural network architecture
   - Handles forward passes
   - Manages the target network
   - Responsible for network updates

2. **DoubleCNNTrainer**:
   - Manages the training process
   - Handles data loading and preprocessing
   - Implements the training loop
   - Manages model checkpoints and logging

This separation makes the code:
- More maintainable
- Easier to test
- More modular
- Easier to extend

## Why 2D CNN?

The implementation uses 2D CNNs because:
1. **Feature Extraction**: CNNs excel at learning spatial patterns
2. **Dimensionality**: Input data is reshaped into 2D images
3. **Efficiency**: 2D convolutions are computationally efficient
4. **Pattern Recognition**: Better at capturing local patterns in the data

## Performance Monitoring

The implementation includes comprehensive monitoring:
- Training loss and accuracy
- Validation metrics
- Learning rate adjustments
- Early stopping
- Model checkpointing
- Visualization tools

## Requirements

- PyTorch
- NumPy
- scikit-learn
- pandas
- matplotlib (for visualization)

## Directory Structure

```
.
├── double_dqn_cnn.py          # Main model implementation
├── train_double_dqn_cnn.py    # Training script
├── Makefile                   # Build commands
├── requirements.txt           # Dependencies
└── README.md                  # This file
```
