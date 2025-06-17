import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from model_utils import ModelVisualizer, ModelLogger
from tracking_utils import ModelTracker
from datetime import datetime
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LAMBDA = 0.01
EPOCHS = 50
BATCH_SIZE = 64
TRAIN_SPLIT_PERCENT = 0.8
EARLY_STOPPING_PATIENCE = 5

class DoubleCNNModel(nn.Module):
    def __init__(self, input_channels, height, width, output_dim):
        super(DoubleCNNModel, self).__init__()
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
        
        # Target network (for Double DQN)
        self.target_conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.target_bn1 = nn.BatchNorm2d(16)
        self.target_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.target_bn2 = nn.BatchNorm2d(32)
        self.target_fc1 = nn.Linear(self.conv_output_size, 64)
        self.target_bn_fc1 = nn.BatchNorm1d(64)
        self.target_fc2 = nn.Linear(64, output_dim)
        
        # Initialize target network with same weights as main network
        self.update_target_network()
    
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
    
    def target_forward(self, x):
        batch_size = x.size(0)
        
        # Reshape and normalize input
        if len(x.shape) == 2:
            x = x.view(batch_size, self.input_channels, self.height, self.width)
        x = self.input_bn(x)
        
        # First conv block
        x = F.relu(self.target_bn1(self.target_conv1(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        
        # Second conv block
        x = F.relu(self.target_bn2(self.target_conv2(x)))
        x = self.dropout2d(x)
        
        # Fully connected layers
        x = x.view(batch_size, -1)
        x = F.relu(self.target_bn_fc1(self.target_fc1(x)))
        x = self.dropout(x)
        x = self.target_fc2(x)
        
        return x
    
    def update_target_network(self, tau=0.005):
        """Soft update of target network"""
        for target_param, param in zip(self.target_conv1.parameters(), self.conv1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_bn1.parameters(), self.bn1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_conv2.parameters(), self.conv2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_bn2.parameters(), self.bn2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_fc1.parameters(), self.fc1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_bn_fc1.parameters(), self.bn_fc1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_fc2.parameters(), self.fc2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

class DoubleCNNTrainer:
    def __init__(self, model, learning_rate=LEARNING_RATE, device=device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        # Initialize tracker
        self.tracker = ModelTracker('double_dqn_cnn')
        self.epsilon = EPSILON_START
        
    def compute_class_weights(self, labels):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.FloatTensor(class_weights).to(self.device)
    
    def update_epsilon(self):
        """Update epsilon using exponential decay"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def train_step(self, inputs, targets, class_weights=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        inputs = torch.FloatTensor(inputs).to(self.device)
        targets = torch.LongTensor(targets).to(self.device)
        
        # Forward pass through main network
        outputs = self.model(inputs)
        
        # Get actions from main network
        actions = torch.argmax(outputs, dim=1)
        
        # Get Q-values from target network
        with torch.no_grad():
            target_outputs = self.model.target_forward(inputs)
            target_q_values = target_outputs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
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
        
        # Update target network
        self.model.update_target_network()
        
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == targets).float().mean().item()
        
        # Log metrics
        metrics = {
            'train_loss': loss.item(),
            'train_accuracy': accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'epsilon': self.epsilon
        }
        self.tracker.log_metrics(metrics, epoch=0, phase='train')
        
        return loss.item(), accuracy
    
    @torch.no_grad()
    def evaluate(self, inputs, targets):
        self.model.eval()
        
        # Move data to device
        inputs = torch.FloatTensor(inputs).to(self.device)
        targets = torch.LongTensor(targets).to(self.device)
        
        # Forward pass through main network
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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['accuracy']

    def run_training(self, train_loader, val_loader=None, epochs=EPOCHS, save_dir='checkpoints', 
                    early_stopping_patience=EARLY_STOPPING_PATIENCE, class_weights=None):
        """
        Run the complete training process
        """
        best_accuracy = 0.0
        patience_counter = 0
        
        # Log model summary
        model_summary = {
            'model_name': 'double_dqn_cnn',
            'input_channels': self.model.input_channels,
            'height': self.model.height,
            'width': self.model.width,
            'output_dim': self.model.fc2.out_features,
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        self.tracker.save_summary(model_summary)
        
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
            
            # Calculate average metrics for the epoch
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
            
            # Update epsilon
            self.update_epsilon()
            
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
                
                # Log validation metrics
                val_metrics = {
                    'val_loss': avg_loss,  # Using training loss as approximation
                    'val_accuracy': val_accuracy,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1
                }
                self.tracker.log_metrics(val_metrics, epoch=epoch, phase='val')
                
                # Update learning rate based on validation accuracy
                self.scheduler.step(val_accuracy)
                
                # Plot metrics every few epochs
                if (epoch + 1) % 2 == 0:
                    self.tracker.plot_training_curves()
                    self.tracker.plot_learning_rate_and_epsilon()
                    self.tracker.plot_confusion_matrix(
                        val_targets,
                        val_predictions,
                        class_names=[str(i) for i in range(len(np.unique(val_targets)))]
                    )
                
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
            
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f} Avg Accuracy: {avg_accuracy:.4f}")
            print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Current epsilon: {self.epsilon:.4f}")
        
        return self.tracker.metrics_history['train_loss'], self.tracker.metrics_history['train_accuracy'], best_accuracy 