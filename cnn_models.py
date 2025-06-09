import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNModel(nn.Module):
    def __init__(self, input_channels, height, width, output_dim):
        super(CNNModel, self).__init__()
        self.input_channels = input_channels
        self.height = height
        self.width = width
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm2d(input_channels)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        conv_width = width // 2  # After one pooling layer
        conv_height = height // 2
        self.conv_output_size = 32 * conv_width * conv_height
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape and normalize input
        if len(x.shape) == 2:
            x = x.view(batch_size, self.input_channels, self.height, self.width)
        x = self.input_bn(x)
        
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2d(x)
        
        # Fully connected layers
        x = x.view(batch_size, -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CNNTrainer:
    def __init__(self, model, learning_rate=0.001, device=device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)
        
    def compute_class_weights(self, labels):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.FloatTensor(class_weights).to(self.device)
    
    def train_step(self, inputs, targets, class_weights=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        inputs = torch.FloatTensor(inputs).to(self.device)
        targets = torch.LongTensor(targets).to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Loss calculation
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == targets).float().mean().item()
        
        return loss.item(), accuracy
    
    @torch.no_grad()
    def evaluate(self, inputs, targets):
        self.model.eval()
        
        # Move data to device
        inputs = torch.FloatTensor(inputs).to(self.device)
        targets = torch.LongTensor(targets).to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        predictions = torch.argmax(outputs, dim=1)
        
        # Calculate metrics
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        accuracy = (predictions == targets).float().mean().item()
        
        # Calculate additional metrics if more than one sample
        if len(targets_np) > 1:
            precision = precision_score(targets_np, predictions_np, average='weighted', zero_division=0)
            recall = recall_score(targets_np, predictions_np, average='weighted', zero_division=0)
            f1 = f1_score(targets_np, predictions_np, average='weighted', zero_division=0)
        else:
            precision = recall = f1 = accuracy
        
        return predictions_np, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_model(self, path, epoch, accuracy):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['accuracy']

    def run_training(self, train_loader, val_loader=None, epochs=10, save_dir='checkpoints', 
                    early_stopping_patience=5, class_weights=None):
        """
        Run the complete training process
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints and plots
            early_stopping_patience: Number of epochs to wait before early stopping
            class_weights: Optional class weights for imbalanced datasets
        """
        os.makedirs(save_dir, exist_ok=True)
        best_accuracy = 0.0
        patience_counter = 0
        training_losses = []
        training_accuracies = []
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Using device: {self.device}")
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_accuracies = []
            
            # Training phase
            self.model.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                loss, accuracy = self.train_step(inputs, targets, class_weights)
                epoch_losses.append(loss)
                epoch_accuracies.append(accuracy)
                training_losses.append(loss)
                training_accuracies.append(accuracy)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss:.4f} Accuracy: {accuracy:.4f}")
            
            # Calculate average metrics for the epoch
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
            
            # Validation phase
            if val_loader is not None:
                val_predictions = []
                val_targets = []
                val_metrics_list = []
                self.model.eval()
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        predictions, metrics = self.evaluate(inputs, targets)
                        val_predictions.extend(predictions)
                        val_targets.extend(targets.numpy())
                        val_metrics_list.append(metrics)
                
                # Calculate average validation metrics
                val_accuracy = np.mean([m['accuracy'] for m in val_metrics_list])
                val_precision = np.mean([m['precision'] for m in val_metrics_list])
                val_recall = np.mean([m['recall'] for m in val_metrics_list])
                val_f1 = np.mean([m['f1'] for m in val_metrics_list])
                
                print(f"Epoch {epoch+1} - Validation Metrics:")
                print(f"  Accuracy: {val_accuracy:.4f}")
                print(f"  Precision: {val_precision:.4f}")
                print(f"  Recall: {val_recall:.4f}")
                print(f"  F1 Score: {val_f1:.4f}")
                
                # Plot metrics every few epochs
                if (epoch + 1) % 2 == 0:
                    # Plot confusion matrix
                    class_names = [str(i) for i in range(len(np.unique(val_targets)))]
                    confusion_matrix_path = ModelVisualizer.plot_confusion_matrix(
                        val_targets,
                        val_predictions,
                        class_names,
                        save_dir=os.path.join(save_dir, 'plots')
                    )
                    print(f"Confusion matrix saved to: {confusion_matrix_path}")
                    
                    # Plot class-wise metrics
                    metrics_path, class_metrics = ModelVisualizer.plot_class_metrics(
                        val_targets,
                        val_predictions,
                        class_names,
                        save_dir=os.path.join(save_dir, 'plots')
                    )
                    print(f"Class-wise metrics plot saved to: {metrics_path}")
                
                # Learning rate scheduling
                self.scheduler.step(val_accuracy)
                
                # Early stopping check
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    patience_counter = 0
                    # Save best model
                    self.save_model(
                        os.path.join(save_dir, f'best_model_epoch_{epoch+1}.pth'),
                        epoch,
                        val_accuracy
                    )
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Plot training metrics every few epochs
            if (epoch + 1) % 2 == 0:
                plot_path = ModelVisualizer.plot_training_metrics(
                    training_losses, 
                    training_accuracies,
                    save_dir=os.path.join(save_dir, 'plots')
                )
                print(f"Training plots saved to {plot_path}")
            
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f} Avg Accuracy: {avg_accuracy:.4f}")
        
        return training_losses, training_accuracies, best_accuracy

class ModelVisualizer:
    @staticmethod
    def setup_plot_style():
        plt.style.use('bmh')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
    
    @staticmethod
    def plot_training_metrics(losses, accuracies, save_dir='logs'):
        ModelVisualizer.setup_plot_style()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss with moving average
        window_size = 50
        losses_smooth = np.convolve(np.array(losses), np.ones(window_size)/window_size, mode='valid')
        ax1.plot(losses, label='Training Loss', color='red', alpha=0.3, linewidth=1)
        ax1.plot(np.arange(window_size-1, len(losses)), losses_smooth, 
                label='Moving Average', color='red', linewidth=2)
        ax1.set_title('Training Loss Over Time')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy with moving average
        accuracies_smooth = np.convolve(accuracies, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(accuracies, label='Training Accuracy', color='blue', alpha=0.3, linewidth=1)
        ax2.plot(np.arange(window_size-1, len(accuracies)), accuracies_smooth,
                label='Moving Average', color='blue', linewidth=2)
        ax2.set_title('Training Accuracy Over Time')
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        # Save plot
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'{save_dir}/training_metrics_{timestamp}.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, save_dir='logs'):
        ModelVisualizer.setup_plot_style()
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        
        # Add value annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'{save_dir}/confusion_matrix_{timestamp}.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

    @staticmethod
    def plot_class_metrics(y_true, y_pred, classes, save_dir='logs'):
        """Plot precision, recall, and F1-score for each class."""
        ModelVisualizer.setup_plot_style()
        
        # Calculate metrics for each class
        precisions = precision_score(y_true, y_pred, labels=range(len(classes)), average=None, zero_division=0)
        recalls = recall_score(y_true, y_pred, labels=range(len(classes)), average=None, zero_division=0)
        f1_scores = f1_score(y_true, y_pred, labels=range(len(classes)), average=None, zero_division=0)
        
        # Create bar plot
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precisions, width, label='Precision', color='skyblue')
        ax.bar(x, recalls, width, label='Recall', color='lightgreen')
        ax.bar(x + width, f1_scores, width, label='F1-score', color='salmon')
        
        # Customize plot
        ax.set_ylabel('Score')
        ax.set_title('Class-wise Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
        
        for container in ax.containers:
            add_value_labels(container)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'{save_dir}/class_metrics_{timestamp}.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path, {
            'precision': precisions,
            'recall': recalls,
            'f1': f1_scores
        }

# Example usage
if __name__ == "__main__":
    import torch.utils.data as data
    from torchvision import datasets, transforms
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Example parameters
    input_channels = 1
    height = width = 28  # MNIST image size
    output_dim = 310     # MNIST has 10 classes
    batch_size = 32
    epochs = 10
    
    # Create data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST dataset as an example
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model and trainer
    model = CNNModel(input_channels, height, width, output_dim)
    trainer = CNNTrainer(model)
    
    # Train the model
    losses, accuracies, best_acc = trainer.run_training(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        save_dir='mnist_training',
        early_stopping_patience=3
    )
    
    print(f"\nTraining completed! Best accuracy: {best_acc:.4f}")
    
    # Final evaluation
    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            predictions, _ = trainer.evaluate(inputs, targets)
            test_predictions.extend(predictions)
            test_targets.extend(targets.numpy())
    
    final_accuracy = np.mean(np.array(test_predictions) == np.array(test_targets))
    print(f"Final Test Accuracy: {final_accuracy:.4f}")

    # Plot confusion matrix
    class_names = [str(i) for i in range(10)]  # MNIST has 10 classes (0-9)
    confusion_matrix_path = ModelVisualizer.plot_confusion_matrix(
        test_targets, test_predictions, class_names, save_dir='mnist_training/plots'
    )
    print(f"Confusion matrix saved to: {confusion_matrix_path}")

    # Plot class metrics
    class_metrics_path, class_metrics = ModelVisualizer.plot_class_metrics(
        test_targets, test_predictions, class_names, save_dir='mnist_training/plots'
    )
    print(f"Class metrics plot saved to: {class_metrics_path}")
    print("Class-wise metrics:", class_metrics)