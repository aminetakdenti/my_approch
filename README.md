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
