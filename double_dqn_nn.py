import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score
from model_utils import ModelVisualizer, ModelLogger
from tracking_utils import ModelTracker
from typing import List, Tuple, Dict, Optional, Union
from datetime import datetime
import os
from dqn_nn import DQNModel, DataLoader, device, LEARNING_RATE, EPSILON_START, EPSILON_END, EPSILON_DECAY, LAMBDA, EPOCHS, BATCH_SIZE, TRAIN_SPLIT_PERCENT, EARLY_STOPPING_PATIENCE

class DoubleDQNModel(DQNModel):
    """
    Double DQN model that inherits from the base DQN model.
    The architecture remains the same, but the training process is modified to implement Double Q-learning.
    
    The key difference is in how Q-values are computed during training:
    1. The online network selects actions
    2. The target network evaluates the selected actions
    
    This helps reduce overestimation of Q-values and provides more stable training.
    """
    pass

class DoubleDeepQNet:
    """
    Implementation of Double Deep Q-Network (Double DQN).
    
    Double DQN is an improvement over standard DQN that helps reduce overestimation
    of Q-values by decoupling the action selection from the Q-value estimation.
    
    Attributes:
        data (DataLoader): Data loader containing training and testing data
        input_dim (int): Dimension of input features
        output_dim (int): Number of possible actions
        epsilon (float): Exploration rate for epsilon-greedy policy
        q_network (DoubleDQNModel): Online network for action selection
        target_network (DoubleDQNModel): Target network for Q-value estimation
        optimizer (torch.optim.Optimizer): Optimizer for the online network
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        criterion (nn.Module): Loss function
        training_history (Dict): Dictionary to store training metrics
    """

    def __init__(self, dataset: DataLoader):
        """
        Initialize the Double DQN model.

        Args:
            dataset (DataLoader): Data loader containing training and testing data
        """
        self.data = dataset
        self.input_dim = dataset.feature_dim
        self.output_dim = dataset.actions_classes
        self.epsilon = EPSILON_START
        
        # Initialize tracker
        self.tracker = ModelTracker('double_dqn_nn')
        
        print("here shape: ", self.input_dim, self.output_dim)
        self.model = self.create_model()  # Main DQN model
        self.target_model = self.create_model()  # Target DQN model
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=True
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.target_update_counter = 0

    def create_model(self) -> DoubleDQNModel:
        """
        Create a new Double DQN model.

        Returns:
            DoubleDQNModel: A new instance of the Double DQN model
        """
        return DoubleDQNModel(self.input_dim, self.output_dim).to(device)

    def update_target_network(self) -> None:
        """
        Update target network weights with online network weights.
        This is called periodically during training to maintain a stable target.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def load_model(self, path: str) -> None:
        """
        Load model weights from a file.

        Args:
            path (str): Path to the saved model weights
        """
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.update_target_network()

    def save_model(self, path: str) -> None:
        """
        Save model weights to a file.

        Args:
            path (str): Path where to save the model weights
        """
        torch.save(self.model.state_dict(), path)

    def summary(self) -> None:
        """
        Print a summary of the model architecture and parameters.
        """
        print("Online Network:")
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

    def get_reward(self, predicted_actions: np.ndarray, optimal_actions: np.ndarray) -> np.ndarray:
        """
        Calculate rewards based on whether predicted actions match optimal actions.

        Args:
            predicted_actions (np.ndarray): Actions predicted by the model
            optimal_actions (np.ndarray): Optimal actions from the dataset

        Returns:
            np.ndarray: Binary reward vector (1 for correct predictions, 0 for incorrect)
        """
        predicted_actions = np.asarray(predicted_actions).reshape(-1)
        optimal_actions = np.asarray(optimal_actions).reshape(-1)
        reward_vector = np.equal(predicted_actions, optimal_actions).astype(np.uint32)
        return reward_vector

    def greedy(self, actions_values_vec: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Implement epsilon-greedy action selection.

        Args:
            actions_values_vec (np.ndarray): Q-values for all actions
            epsilon (float): Exploration rate

        Returns:
            np.ndarray: Selected actions
        """
        random.seed(a=None, version=2)
        num_in_curr_batch = actions_values_vec.shape[0]
        selections = []
        for i in range(num_in_curr_batch):
            p = random.uniform(0.0, 1.0)
            if p < epsilon:
                curr_actions_values = actions_values_vec[i, :].reshape(-1)
                selections.append(np.argmax(curr_actions_values))
            else:
                random_selection = np.random.randint(low=0, high=self.data.actions_classes)
                selections.append(random_selection)
        return np.asarray(selections)

    def process_batch(self, batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a batch of data into states, actions, and next states.

        Args:
            batch (np.ndarray): Batch of data

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Current states, optimal actions, and next states
        """
        current_states = batch[:, :self.data.feature_dim]
        optimal_actions = batch[:, self.data.feature_dim].astype(int)
        next_states = batch[:, self.data.feature_dim + 1:]
        return current_states, optimal_actions, next_states

    def update_epsilon(self) -> None:
        """
        Update the exploration rate (epsilon) using exponential decay.
        """
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def train(self, save_path: Optional[str] = None, plot_dir: str = 'logs') -> Tuple[List[float], List[float]]:
        """
        Train the Double DQN model.

        Args:
            save_path (Optional[str]): Path to save the best model
            plot_dir (str): Directory to save training plots

        Returns:
            Tuple[List[float], List[float]]: Lists of losses and accuracies during training
        """
        try:
            best_accuracy = 0
            patience_counter = 0
            batches = self.data.train_batches
            
            # Log model summary
            model_summary = {
                'model_name': 'double_dqn_nn',
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
            self.tracker.save_summary(model_summary)
            
            print(f"Starting training for {EPOCHS} epochs...")
            print(f"Using device: {device}")
            
            self.model.train()
            self.target_model.train()
            
            for epoch in range(EPOCHS):
                epoch_losses = []
                epoch_accuracies = []
                
                for batch_idx, batch in enumerate(batches):
                    current_states, optimal_actions, next_states = self.process_batch(batch)
                    
                    current_states_tensor = torch.FloatTensor(current_states).to(device)
                    next_states_tensor = torch.FloatTensor(next_states).to(device)
                    optimal_actions_tensor = torch.LongTensor(optimal_actions).to(device)
                    
                    # Get Q-values from main network
                    estimated_qs_vec_t = self.model(current_states_tensor)
                    estimated_qs_vec_t_np = estimated_qs_vec_t.detach().cpu().numpy()
                    predicted_actions_t = self.greedy(estimated_qs_vec_t_np, epsilon=self.epsilon)
                    rewards_t = self.get_reward(predicted_actions_t, optimal_actions)
                    
                    with torch.no_grad():
                        # Get Q-values from target network
                        estimated_qs_vec_t_plus_one = self.target_model(next_states_tensor)
                        estimated_qs_vec_t_plus_one_np = estimated_qs_vec_t_plus_one.cpu().numpy()
                        predicted_actions_t_plus_one = self.greedy(estimated_qs_vec_t_plus_one_np, epsilon=1.0)
                        all_rows_idx = np.arange(estimated_qs_vec_t_plus_one_np.shape[0])
                        q_cap_t_plus_one = estimated_qs_vec_t_plus_one_np[all_rows_idx, predicted_actions_t_plus_one]
                    
                    qref = np.zeros_like(estimated_qs_vec_t_np)
                    qref[all_rows_idx, optimal_actions] = 1
                    qref[all_rows_idx, predicted_actions_t] += LAMBDA * q_cap_t_plus_one
                    qref_softmax = np.zeros_like(qref)
                    qref_softmax[all_rows_idx, qref.argmax(1)] = 1
                    
                    qref_softmax_tensor = torch.FloatTensor(qref_softmax).to(device)
                    
                    self.optimizer.zero_grad()
                    loss = self.criterion(estimated_qs_vec_t, qref_softmax_tensor)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    with torch.no_grad():
                        predictions = torch.argmax(estimated_qs_vec_t, dim=1)
                        targets = torch.argmax(qref_softmax_tensor, dim=1)
                        accuracy = (predictions == targets).float().mean().item()
                    
                    epoch_losses.append(loss.item())
                    epoch_accuracies.append(accuracy)
                    
                    # Log metrics
                    metrics = {
                        'train_loss': loss.item(),
                        'train_accuracy': accuracy,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epsilon': self.epsilon
                    }
                    self.tracker.log_metrics(metrics, epoch=epoch, phase='train')
                    
                    # Update target network
                    self.target_update_counter += 1
                    if self.target_update_counter % TARGET_UPDATE_FREQ == 0:
                        self.update_target_network()
                
                # Update epsilon after each epoch
                self.update_epsilon()
                
                # Calculate epoch metrics
                avg_epoch_accuracy = np.mean(epoch_accuracies)
                avg_epoch_loss = np.mean(epoch_losses)
                
                # Update learning rate based on accuracy
                self.scheduler.step(avg_epoch_accuracy)
                
                # Log validation metrics
                val_metrics = {
                    'val_loss': avg_epoch_loss,
                    'val_accuracy': avg_epoch_accuracy
                }
                self.tracker.log_metrics(val_metrics, epoch=epoch, phase='val')
                
                # Plot metrics every few epochs
                if (epoch + 1) % 2 == 0:
                    self.tracker.plot_training_curves()
                    self.tracker.plot_learning_rate_and_epsilon()
                
                # Early stopping check
                if avg_epoch_accuracy > best_accuracy:
                    best_accuracy = avg_epoch_accuracy
                    patience_counter = 0
                    if save_path is not None:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                
                print(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f} Avg Accuracy: {avg_epoch_accuracy:.4f}")
                print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                print(f"Current epsilon: {self.epsilon:.4f}")
            
            return self.tracker.metrics_history['train_loss'], self.tracker.metrics_history['train_accuracy']
            
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            raise

    def test(self) -> Tuple[float, float, float, float]:
        """
        Evaluate the trained model on the test set.

        Returns:
            Tuple[float, float, float, float]: Accuracy, precision, recall, and F1 score
        """
        batches = self.data.test_batches
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        self.target_model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(batches):
                current_states, optimal_actions, _ = self.process_batch(batch)
                current_states_tensor = torch.FloatTensor(current_states).to(device)
                
                estimated_qs_vec = self.model(current_states_tensor)
                estimated_qs_vec_np = estimated_qs_vec.cpu().numpy()
                predicted_actions = self.greedy(estimated_qs_vec_np, epsilon=1.0).squeeze()
                
                all_predictions.extend(predicted_actions)
                all_targets.extend(optimal_actions)
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        # Log test metrics
        test_metrics = {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        }
        self.tracker.log_metrics(test_metrics, epoch=0, phase='test')
        
        # Plot confusion matrix
        self.tracker.plot_confusion_matrix(
            all_targets,
            all_predictions,
            class_names=[str(i) for i in range(self.data.actions_classes)]
        )
        
        return accuracy, precision, recall, f1

    def predict(self, states: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            states (Union[np.ndarray, torch.Tensor]): Input states

        Returns:
            np.ndarray: Predicted actions
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(states, np.ndarray):
                states_tensor = torch.FloatTensor(states).to(device)
            else:
                states_tensor = states.to(device)

            estimated_qs = self.model(states_tensor)
            estimated_qs_np = estimated_qs.cpu().numpy()
            predicted_action = self.greedy(estimated_qs_np, epsilon=1.0).squeeze()
        return predicted_action
