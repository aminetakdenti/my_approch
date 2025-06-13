import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
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

class CNNDQNModel(nn.Module):
    def __init__(self, input_channels, height, width, output_dim):
        super(CNNDQNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size after convolutions
        conv_out_size = self._get_conv_out_size(input_channels, height, width)
        
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, output_dim)
        
    def _get_conv_out_size(self, channels, height, width):
        # Helper to calculate output size after convolutions
        x = torch.zeros(1, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()))
        
    def forward(self, x):
        # Reshape input if needed: (batch, features) -> (batch, channels, height, width)
        if len(x.shape) == 2:
            # Assuming square images, adjust as needed
            side = int(np.sqrt(x.shape[1]))
            x = x.view(-1, 1, side, side)
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class DataLoader:
    def __init__(self, csv_filepath, batch_size):
        self.df_samples = pd.read_csv(csv_filepath)  # Create a pandas dataframe
        self.numpy_samples = self.df_samples.to_numpy()

        self.states_features = self.numpy_samples[
            :, 1 : self.numpy_samples.shape[1] - 1
        ]  # Take the feature values for states, also ignore first column for IDs
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

    def plot_training_metrics(self, save_path: str = None):
        """
        Plot training metrics including loss, accuracy, epsilon, and learning rate.
        
        Args:
            save_path (str, optional): Directory to save the plot. If None, will use 'plots' directory.
        """
        # Use a default style
        plt.style.use('default')
        
        # Create figure with a white background
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.patch.set_facecolor('white')
        
        # Plot Loss
        ax1.plot(self.training_history['losses'], label='Training Loss', color='blue', linewidth=2)
        ax1.set_title('Training Loss Over Time', fontsize=12, pad=10)
        ax1.set_xlabel('Batch', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot Accuracy
        ax2.plot(self.training_history['accuracies'], label='Training Accuracy', color='green', linewidth=2)
        ax2.set_title('Training Accuracy Over Time', fontsize=12, pad=10)
        ax2.set_xlabel('Batch', fontsize=10)
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot Epsilon
        ax3.plot(self.training_history['epsilons'], label='Epsilon', color='red', linewidth=2)
        ax3.set_title('Epsilon Value Over Time', fontsize=12, pad=10)
        ax3.set_xlabel('Epoch', fontsize=10)
        ax3.set_ylabel('Epsilon', fontsize=10)
        ax3.legend(fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Plot Learning Rate
        ax4.plot(self.training_history['learning_rates'], label='Learning Rate', color='purple', linewidth=2)
        ax4.set_title('Learning Rate Over Time', fontsize=12, pad=10)
        ax4.set_xlabel('Epoch', fontsize=10)
        ax4.set_ylabel('Learning Rate', fontsize=10)
        ax4.legend(fontsize=10)
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and spacing
        plt.tight_layout(pad=3.0)
        
        # Create plots directory if it doesn't exist
        if save_path is None:
            save_path = 'plots'
        os.makedirs(save_path, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(save_path, f'training_metrics_{timestamp}.png')
        
        # Save the plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training metrics plot saved to: {plot_path}")
        return plot_path

    def train(self, save_path=None, plot_dir='plots'):
        try:
            losses = []
            accuracies = []
            best_accuracy = 0
            patience_counter = 0
            batches = self.data.train_batches

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

                    print(" -------------------------------------------------- ")
                    print(f"In epoch {epoch + 1}/{EPOCHS}, batch {batch_idx + 1}/{self.data.n_train_batches}:")
                    print(f"Accuracy: {accuracy:.4f}")
                    print(f"Loss: {loss.item():.4f}")
                    print(f"Epsilon: {self.epsilon:.4f}")
                    print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                    print(" -------------------------------------------------- ")

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
                    self.plot_training_metrics(plot_dir)

            return losses, accuracies
            
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            raise

    def test(self):
        batches = self.data.test_batches

        accuracy = 0
        f1_score = 0
        precision = 0
        recall = 0

        self.model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for batch_idx, batch in enumerate(batches):  # Looping over the batches
                current_states, optimal_actions, _ = self.process_batch(
                    batch
                )  # Get the data from the batch

                # Convert to tensor
                current_states_tensor = torch.FloatTensor(current_states).to(device)

                estimated_qs_vec = self.model(current_states_tensor)
                estimated_qs_vec_np = estimated_qs_vec.cpu().numpy()
                predicted_actions = self.greedy(
                    estimated_qs_vec_np, epsilon=1.0
                ).squeeze()  # Since we are testing so we need no exploration, we are only greedy now (eps=1.0)

                curr_batch_accuracy = np.mean(
                    np.equal(predicted_actions, optimal_actions).astype(np.uint32)
                )
                curr_batch_precision = precision_score(
                    optimal_actions,
                    predicted_actions,
                    labels=self.data.actions_set,
                    average="weighted",
                    zero_division=1,
                )
                curr_batch_recall = recall_score(
                    optimal_actions,
                    predicted_actions,
                    labels=self.data.actions_set,
                    average="weighted",
                    zero_division=1,
                )
                curr_batch_f1 = 2 * (
                    (curr_batch_precision * curr_batch_recall)
                    / (curr_batch_precision + curr_batch_recall + 1e-7)
                )

                accuracy += curr_batch_accuracy / len(batches)
                precision += curr_batch_precision / len(batches)
                recall += curr_batch_recall / len(batches)
                f1_score += curr_batch_f1 / len(batches)

        print("Finished testing on the testing dataset, now printing metrics.")
        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1 Score: {}".format(f1_score))
        return accuracy, precision, recall, f1_score

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
