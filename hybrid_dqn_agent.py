import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score
import os
from utils.logger import setup_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime

# Hyperparameters
LEARNING_RATE = 0.001
EPSILON = 0.8
LAMBDA = 0.01
EPOCHS = 10
BATCH_SIZE = 64
TRAIN_SPLIT_PERCENT = 0.8
TARGET_UPDATE = 10  # Update target network every N epochs

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add logger setup after the device setup
logger = setup_logger('hybrid_dqn')

class CNNModel(nn.Module):
    def __init__(self, input_channels, height, width, output_dim):
        super(CNNModel, self).__init__()
        self.input_channels = input_channels
        self.height = height
        self.width = width
        
        # Adjust network size for smaller input
        self.input_bn = nn.BatchNorm2d(input_channels)
        
        # Smaller network for 7x7 input
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        conv_width = width // 2  # After one pooling layer
        conv_height = height // 2
        self.conv_output_size = 32 * conv_width * conv_height
        
        # Adjust FC layers based on smaller conv output
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

class SimpleTargetNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleTargetNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

class DataLoader:
    def __init__(self, csv_filepath, batch_size, img_height, img_width, img_channels=1):
        logger.info(f"Loading dataset from {csv_filepath}")
        self.df_samples = pd.read_csv(csv_filepath)
        self.numpy_samples = self.df_samples.to_numpy()
        
        # Extract features and reshape for CNN
        self.states_features = self.numpy_samples[:, 1:-1]  # Exclude ID and action
        self.feature_dim = self.states_features.shape[1]
        
        logger.info(f"Original feature dimension: {self.feature_dim}")
        
        # Calculate padding needed
        self.img_size = int(np.ceil(np.sqrt(self.feature_dim)))
        total_size = self.img_size * self.img_size
        
        if self.feature_dim < total_size:
            padding = total_size - self.feature_dim
            logger.info(f"Padding features with {padding} zeros")
            self.states_features = np.pad(
                self.states_features,
                ((0, 0), (0, padding)),
                mode='constant'
            )
        
        self.actions_labels = self.numpy_samples[:, -1].reshape(-1, 1)
        self.actions_classes = int(np.amax(self.actions_labels) + 1)
        self.actions_set = np.arange(self.actions_classes).tolist()
        
        logger.info(f"Final feature dimension: {self.states_features.shape[1]}")
        logger.info(f"Number of action classes: {self.actions_classes}")
        
        self.train_batches, self.test_batches = self.prepare_batches(
            batch_size, train_split_percent=TRAIN_SPLIT_PERCENT
        )
        logger.info(f"Created {len(self.train_batches)} training batches and {len(self.test_batches)} testing batches")

    def prepare_batches(self, batch_size, train_split_percent):
        states_t = self.states_features[:-1, :].copy()
        states_t_plus_one = self.states_features[1:, :].copy()
        actions_star_t = self.actions_labels[:-1, :].copy()
        
        whole_generic_samples = np.hstack((states_t, actions_star_t, states_t_plus_one))
        np.random.seed(0)
        np.random.shuffle(whole_generic_samples)

        self.n_samples = whole_generic_samples.shape[0]
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        
        split_idx = int(self.n_batches * train_split_percent)
        train_batches = []
        test_batches = []
        
        for i in range(self.n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.n_samples)
            curr_batch = whole_generic_samples[start:end, :]
            if i < split_idx:
                train_batches.append(curr_batch)
            else:
                test_batches.append(curr_batch)

        return train_batches, test_batches

class Visualizer:
    @staticmethod
    def setup_plot_style():
        # Use a built-in style instead of seaborn
        plt.style.use('bmh')  # Alternative modern style
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
    @staticmethod
    def plot_training_metrics(losses, accuracies, save_dir='logs'):
        Visualizer.setup_plot_style()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss with moving average
        window_size = 50
        losses_smooth = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
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
        
        # Adjust layout and save
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'{save_dir}/training_metrics_{timestamp}.png'
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training metrics plot saved to {save_path}")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, save_dir='logs'):
        Visualizer.setup_plot_style()
        
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
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {save_path}")

    @staticmethod
    def plot_class_distribution(y, save_dir='logs'):
        Visualizer.setup_plot_style()
        
        # Count class occurrences
        unique, counts = np.unique(y, return_counts=True)
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique, counts, alpha=0.8, color='skyblue', edgecolor='black')
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        
        # Add percentage labels on top of each bar
        total = sum(counts)
        for bar, count in zip(bars, counts):
            percentage = count / total * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count}\n({percentage:.1f}%)',
                    ha='center', va='bottom')
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'{save_dir}/class_distribution_{timestamp}.png'
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Class distribution plot saved to {save_path}")

class HybridDQN:
    def __init__(self, dataset, img_height, img_width, img_channels=1):
        self.data = dataset
        self.input_dim = dataset.feature_dim
        self.output_dim = dataset.actions_classes
        
        # Calculate dimensions for square image
        self.feature_dim = self.input_dim
        self.img_size = int(np.ceil(np.sqrt(self.feature_dim)))
        self.img_height = self.img_size
        self.img_width = self.img_size
        self.img_channels = img_channels
        
        logger.info(f"Using image size: {self.img_size}x{self.img_size}")
        
        # Create models
        self.cnn_model = CNNModel(
            self.img_channels, 
            self.img_height, 
            self.img_width, 
            self.output_dim
        ).to(device)
        
        self.target_net = SimpleTargetNet(
            self.input_dim, 
            self.output_dim
        ).to(device)
        
        # Calculate class weights for balanced training
        all_actions = np.concatenate([batch[:, self.data.feature_dim].astype(np.int64) 
                                    for batch in self.data.train_batches])
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(all_actions),
            y=all_actions
        )
        self.class_weights = torch.FloatTensor(class_weights).to(device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.cnn_model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.01
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2
        )
        
        # Use weighted cross entropy loss
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.update_counter = 0
        
        # Initialize target network
        self.update_target_network()
        
        logger.info(f"Class weights: {class_weights}")
        logger.info(f"Model architecture:\n{self.cnn_model}")

    def update_target_network(self):
        # Convert CNN features to match simple network input
        with torch.no_grad():
            for target_param, cnn_param in zip(
                self.target_net.parameters(), self.cnn_model.parameters()
            ):
                # Adapt CNN parameters to target network shape
                if target_param.shape == cnn_param.shape:
                    target_param.data.copy_(cnn_param.data)

    def process_batch(self, batch):
        current_states = batch[:, :self.data.feature_dim]
        optimal_actions = batch[:, self.data.feature_dim].astype(np.uint32)
        next_states = batch[:, self.data.feature_dim + 1:]
        return current_states, optimal_actions, next_states

    def greedy(self, actions_values_vec, epsilon):
        selections = []
        for i in range(len(actions_values_vec)):
            if random.random() < epsilon:
                selections.append(np.argmax(actions_values_vec[i]))
            else:
                selections.append(random.randint(0, self.output_dim - 1))
        return np.array(selections)

    def train(self, save_path=None):
        logger.info("Starting training with class balanced loss...")
        losses = []
        accuracies = []
        best_accuracy = 0.0
        patience = 5
        patience_counter = 0
        best_model_state = None
        
        # Plot initial class distribution
        all_actions = np.concatenate([batch[:, self.data.feature_dim].astype(np.int64) 
                                    for batch in self.data.train_batches])
        Visualizer.plot_class_distribution(all_actions)

        for epoch in range(EPOCHS):
            epoch_losses = []
            epoch_accuracies = []
            self.cnn_model.train()
            self.target_net.eval()

            for batch_idx, batch in enumerate(self.data.train_batches):
                current_states, optimal_actions, next_states = self.process_batch(batch)
                
                # Convert to tensors
                current_states_tensor = torch.FloatTensor(current_states).to(device)
                next_states_tensor = torch.FloatTensor(next_states).to(device)
                optimal_actions_tensor = torch.LongTensor(optimal_actions).to(device)
                
                # Forward pass
                q_values = self.cnn_model(current_states_tensor)
                
                # Calculate loss with class weights
                loss = self.criterion(q_values, optimal_actions_tensor)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), 1.0)
                self.optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    predictions = torch.argmax(q_values, dim=1)
                    accuracy = (predictions == optimal_actions_tensor).float().mean().item()
                
                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy)
                losses.append(loss.item())
                accuracies.append(accuracy)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(self.data.train_batches)}")
                    logger.info(f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
            
            # Plot training metrics after each epoch
            if (epoch + 1) % 2 == 0:  # Plot every 2 epochs
                Visualizer.plot_training_metrics(losses, accuracies)
            
            # Evaluate and update model
            val_accuracy = self.validate()
            self.scheduler.step(val_accuracy)
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.cnn_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': best_accuracy,
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            if self.update_counter % TARGET_UPDATE == 0:
                self.update_target_network()
            
            self.update_counter += 1
        
        # Save best model
        if save_path and best_model_state:
            logger.info(f"Saving best model with accuracy {best_accuracy:.4f}")
            torch.save(best_model_state, save_path)
        
        # Final plots
        Visualizer.plot_training_metrics(losses, accuracies)
        
        return losses, accuracies

    def validate(self):
        """Run validation on a portion of training data"""
        self.cnn_model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.data.test_batches[:len(self.data.test_batches)//2]:
                current_states, optimal_actions, _ = self.process_batch(batch)
                current_states_tensor = torch.FloatTensor(current_states).to(device)
                optimal_actions_tensor = torch.LongTensor(optimal_actions).to(device)
                
                outputs = self.cnn_model(current_states_tensor)
                predictions = torch.argmax(outputs, dim=1)
                
                total_correct += (predictions == optimal_actions_tensor).sum().item()
                total_samples += optimal_actions_tensor.size(0)
        
        accuracy = total_correct / total_samples
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        return accuracy

    def test(self):
        logger.info("Starting model evaluation...")
        self.cnn_model.eval()
        self.target_net.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data.test_batches):
                current_states, optimal_actions, _ = self.process_batch(batch)
                current_states_tensor = torch.FloatTensor(current_states).to(device)
                
                outputs = self.cnn_model(current_states_tensor)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(optimal_actions)
                
                if (batch_idx + 1) % 5 == 0:
                    logger.info(f"Processed {batch_idx+1}/{len(self.data.test_batches)} test batches")
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        accuracy = np.mean(all_predictions == all_targets)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=1)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=1)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        # Log results
        logger.info("\nTest Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Plot confusion matrix
        class_names = [f"Class {i}" for i in range(self.output_dim)]
        Visualizer.plot_confusion_matrix(all_targets, all_predictions, class_names)
        
        # Plot class distribution for test set
        Visualizer.plot_class_distribution(all_targets)
        
        logger.info("Evaluation completed!")
        return accuracy, precision, recall, f1

    def predict(self, state):
        self.cnn_model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            q_values = self.cnn_model(state_tensor)
            action = self.greedy(q_values.cpu().numpy(), epsilon=1.0)
        return action

# Example usage
if __name__ == "__main__":
    # Calculate image dimensions based on feature size
    dataset = DataLoader(
        "data/dataset.csv",
        BATCH_SIZE,
        None,  # These will be calculated automatically
        None,
        img_channels=1
    )
    
    # Create and train model using the calculated dimensions
    dqn = HybridDQN(
        dataset,
        dataset.img_size,  # Use the calculated size
        dataset.img_size,
        img_channels=1
    )
    
    losses, accuracies = dqn.train()
    accuracy, precision, recall, f1 = dqn.test() 