import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score
from model_utils import ModelVisualizer, ModelLogger
from typing import List, Tuple
from datetime import datetime
import os

# Hyperparameters
LEARNING_RATE = 0.001  # Reduced learning rate for better stability
EPSILON_START = 1.0  # Start with full exploration
EPSILON_END = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate for epsilon
LAMBDA = 0.01  # Discount factor for loss calculation
EPOCHS = 50  # Increased number of epochs
BATCH_SIZE = 64  # Smaller batch size for better generalization
TRAIN_SPLIT_PERCENT = 0.8  # Percentage of the data for training, rest for testing
EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait before early stopping

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNModel, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x

class DataLoader:
    def __init__(self, csv_filepath, batch_size):
        self.df_samples = pd.read_csv(csv_filepath)  # Create a pandas dataframe
        self.numpy_samples = self.df_samples.to_numpy()

        self.states_features = self.numpy_samples[
            :, 0 : self.numpy_samples.shape[1] - 1
        ]  # Take the feature values for states
        self.feature_dim = self.states_features.shape[1]
        self.actions_labels = self.numpy_samples[:, -1].reshape(
            -1, 1
        )  # The action labels separated from the labels
        self.actions_classes = int(
            np.amax(self.actions_labels) + 1
        )  # Number of different action classes to set the output layer dimensions (+1 bec starts at zero)
        self.actions_set = np.arange(
            self.actions_classes
        ).tolist()  # Set of all possible actions
        self.train_batches, self.test_batches = self.prepare_batches(
            batch_size, train_split_percent=TRAIN_SPLIT_PERCENT
        )
        print(
            "Dataset successfully loaded with {} training batches, and {} testing batches with {} batch size.".format(
                len(self.train_batches), len(self.test_batches), batch_size
            )
        )

    def prepare_batches(self, batch_size, train_split_percent):
        states_t = self.states_features[:-1, :].copy()  # Considered as S(t)
        states_t_plus_one = self.states_features[1:, :].copy()  # Considered as S(t+1)
        actions_star_t = self.actions_labels[:-1, :].copy()  # Considered as a*(t)
        whole_generic_samples = np.hstack(
            (states_t, actions_star_t, states_t_plus_one)
        )  # Stack the whole dataset as described in the paper
        np.random.seed(
            0
        )  # To shuffle similarly each time, comment to disable this feature
        np.random.shuffle(whole_generic_samples)  # Shuffle the dataset

        self.n_samples = whole_generic_samples.shape[0]
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        train_batches = []  # Empty list to hold the batches of whole data
        test_batches = []
        # Prepare the data into batches
        for i in range(self.n_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            curr_batch = whole_generic_samples[start:end, :]
            if (i / self.n_batches) < train_split_percent:
                train_batches.append(curr_batch)
            else:
                test_batches.append(curr_batch)

        self.n_train_batches = len(train_batches)
        self.n_test_batches = len(test_batches)
        return train_batches, test_batches

    # Function that takes in the 2D arrays of data and converts lo lists of tuples to be compatible with looping while training
    # TODO: Enhancing these function (All the following isn't used)
    def tupelize(self, array):
        list_of_tuples = list(zip(array.T[0], array.T))
        return list_of_tuples

    # Function to get the unique rows representing unique states, returns a numpy array of rows
    def get_unique_rows(self):
        self.unique_rows = np.unique(self.states_features, axis=0)
        return self.unique_rows

    # Get the pandas dataframe for the data, returns a pandas dataframe
    def get_dataframe(self):
        return self.df_samples


# Creating our main class for our DQN
class DeepQNet:

    def __init__(self, dataset):
        self.data = dataset  # Storing the data in our QNet
        self.input_dim = dataset.feature_dim  # State feature dim
        self.output_dim = dataset.actions_classes
        self.epsilon = EPSILON_START
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'epsilons': [],
            'learning_rates': []
        }

        print("here shape: ", self.input_dim, self.output_dim)
        self.model = self.create_model()  # Main DQN model
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # Fixed scheduler configuration
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        self.criterion = nn.CrossEntropyLoss()

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        # Definition of the neural network architecture mentioned in the paper (3 relu feedforward layers)
        model = DQNModel(self.input_dim, self.output_dim).to(device)
        return model

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=device))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    # Prints the model details
    def summary(self):
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

    def get_reward(self, predicted_actions, optimal_actions):
        predicted_actions = np.asarray(predicted_actions).reshape(-1)
        optimal_actions = np.asarray(optimal_actions).reshape(-1)
        reward_vector = np.equal(predicted_actions, optimal_actions).astype(np.uint32)
        return reward_vector

    # Function to implement the epsilon-greedy policy selection, returns the index of the selected action
    def greedy(self, actions_values_vec, epsilon):
        random.seed(a=None, version=2)  # Change the seed of randomization
        num_in_curr_batch = actions_values_vec.shape[0]
        selections = []
        for i in range(num_in_curr_batch):
            p = random.uniform(0.0, 1.0)
            if p < epsilon:
                curr_actions_values = actions_values_vec[i, :].reshape(-1)
                selections.append(np.argmax(curr_actions_values))
            else:
                random_selection = np.random.randint(
                    low=0, high=self.data.actions_classes
                )
                selections.append(random_selection)

        return np.asarray(selections)

    # Function to process the batch and split the S(t), a*(t), and S(t+1)
    def process_batch(self, batch):
        current_states = batch[:, : self.data.feature_dim]
        optimal_actions = batch[:, self.data.feature_dim].astype(
            int
        )  # Ensure integer type
        next_states = batch[:, self.data.feature_dim + 1 :]
        return current_states, optimal_actions, next_states

    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def train(self, save_path=None, plot_dir='logs'):
        try:
            losses = []
            accuracies = []
            best_accuracy = 0
            patience_counter = 0
            batches = self.data.train_batches

            # Log training start
            ModelLogger.log_training_start(EPOCHS, device, "DQN Model")
            ModelLogger.log_model_summary(self.model)

            self.model.train()

            for epoch in range(EPOCHS):
                epoch_losses = []
                epoch_accuracies = []

                for batch_idx, batch in enumerate(batches):
                    current_states, optimal_actions, next_states = self.process_batch(batch)

                    current_states_tensor = torch.FloatTensor(current_states).to(device)
                    next_states_tensor = torch.FloatTensor(next_states).to(device)
                    optimal_actions_tensor = torch.LongTensor(optimal_actions).to(device)

                    estimated_qs_vec_t = self.model(current_states_tensor)
                    estimated_qs_vec_t_np = estimated_qs_vec_t.detach().cpu().numpy()
                    predicted_actions_t = self.greedy(estimated_qs_vec_t_np, epsilon=self.epsilon)
                    rewards_t = self.get_reward(predicted_actions_t, optimal_actions)

                    with torch.no_grad():
                        estimated_qs_vec_t_plus_one = self.model(next_states_tensor)
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

                    # Store metrics for plotting
                    self.training_history['losses'].append(loss.item())
                    self.training_history['accuracies'].append(accuracy)
                    self.training_history['epsilons'].append(self.epsilon)
                    self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

                    # Log batch metrics
                    metrics = {
                        'epochs': EPOCHS,
                        'accuracy': accuracy,
                        'loss': loss.item(),
                        'epsilon': self.epsilon,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }
                    ModelLogger.log_metrics(metrics, epoch, batch_idx, len(batches))

                # Update epsilon after each epoch
                self.update_epsilon()

                # Calculate epoch metrics
                avg_epoch_accuracy = np.mean(epoch_accuracies)
                avg_epoch_loss = np.mean(epoch_losses)
                
                # Update learning rate based on accuracy
                self.scheduler.step(avg_epoch_accuracy)
                
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

                losses.extend(epoch_losses)
                accuracies.extend(epoch_accuracies)

                # Plot training metrics every few epochs
                if (epoch + 1) % 2 == 0:
                    ModelVisualizer.plot_training_metrics(
                        self.training_history,
                        save_dir=plot_dir,
                        prefix='dqn_'
                    )

            # Log training end
            final_metrics = {
                'final_accuracy': best_accuracy,
                'final_loss': np.mean(losses[-len(batches):])
            }
            ModelLogger.log_training_end(best_accuracy, final_metrics)

            return losses, accuracies
            
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            raise

    def test(self):
        batches = self.data.test_batches
        all_predictions = []
        all_targets = []

        self.model.eval()

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

        # Log test results
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        ModelLogger.log_metrics(metrics, 0)

        # Plot confusion matrix and class metrics
        class_names = [str(i) for i in range(self.data.actions_classes)]
        ModelVisualizer.plot_confusion_matrix(
            all_targets,
            all_predictions,
            class_names,
            save_dir='logs',
            prefix='dqn_'
        )
        ModelVisualizer.plot_class_metrics(
            all_targets,
            all_predictions,
            class_names,
            save_dir='logs',
            prefix='dqn_'
        )

        return accuracy, precision, recall, f1

    def predict(self, states):
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
