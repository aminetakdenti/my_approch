import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score
import os

# Hyperparameters
LEARNING_RATE = 0.01  # Gradient-descent learning rate
EPSILON = 0.8  # Epsilon value for the epsilon greedy policy selection
LAMBDA = 0.01  # Discount factor for loss calculation
EPOCHS = 10  # Number of training epochs
BATCH_SIZE = 128  # Default batch size
TRAIN_SPLIT_PERCENT = 0.8  # Percentage of the data for training, rest for testing

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

class CNNDQNModel(nn.Module):
    def __init__(self, input_channels, height, width, output_dim):
        super(CNNDQNModel, self).__init__()
        self.input_channels = input_channels
        self.height = height
        self.width = width
        
        # Use smaller kernels and strides for small images
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
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
            x = x.view(-1, self.input_channels, self.height, self.width)
            
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

    def __init__(self, dataset, use_cnn=True, img_height=None, img_width=None, img_channels=1):
        self.data = dataset  # Storing the data in our QNet
        self.input_dim = dataset.feature_dim  # State feature dim
        self.output_dim = dataset.actions_classes
        self.use_cnn = use_cnn
        
        # Automatically switch to FC if features are too few for CNN
        min_cnn_features = 16  # e.g., 4x4 image minimum
        if self.use_cnn and self.input_dim < min_cnn_features:
            print(f"[WARNING] Feature count ({self.input_dim}) too small for CNN. Using FC model instead.")
            self.use_cnn = False
        
        # CNN parameters
        if self.use_cnn:
            # If no image dimensions provided, try to infer square image
            if img_height is None or img_width is None:
                # Assume square image
                side_length = int(np.sqrt(self.input_dim))
                if side_length * side_length != self.input_dim:
                    # If not perfect square, pad to next square
                    side_length = int(np.ceil(np.sqrt(self.input_dim)))
                    print(f"Warning: Input features ({self.input_dim}) don't form perfect square.")
                    print(f"Padding to {side_length}x{side_length} = {side_length**2} features")
                self.img_height = side_length
                self.img_width = side_length
            else:
                self.img_height = img_height
                self.img_width = img_width
            
            self.img_channels = img_channels
            
            self.model_params = {
                'input_channels': self.img_channels,
                'height': self.img_height,
                'width': self.img_width
            }
        else:
            self.model_params = {
                'input_dim': self.input_dim
            }

        print(f"Model type: {'CNN' if self.use_cnn else 'FC'}")
        print(f"Input shape: {self.input_dim} features")
        if self.use_cnn:
            print(f"CNN input shape: {self.img_channels}x{self.img_height}x{self.img_width}")
        print(f"Output dim: {self.output_dim}")
        
        self.model = self.create_model()  # Main DQN model
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        # Definition of the neural network architecture mentioned in the paper
        if self.use_cnn:
            model = CNNDQNModel(**self.model_params, output_dim=self.output_dim).to(device)
        else:
            model = DQNModel(**self.model_params, output_dim=self.output_dim).to(device)
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

    def pad_input(self, x):
        """Pad input to match expected CNN input size if needed"""
        if self.use_cnn:
            expected_size = self.img_channels * self.img_height * self.img_width
            if x.shape[1] < expected_size:
                # Pad with zeros
                padding = expected_size - x.shape[1]
                x = np.pad(x, ((0, 0), (0, padding)), mode='constant', constant_values=0)
            elif x.shape[1] > expected_size:
                # Truncate
                x = x[:, :expected_size]
        return x

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
        
        # Pad inputs if using CNN
        if self.use_cnn:
            current_states = self.pad_input(current_states)
            next_states = self.pad_input(next_states)
            
        return current_states, optimal_actions, next_states

    def train(self, save_path=None):

        losses = []
        accuracies = []
        batches = self.data.train_batches  # Get the batches

        self.model.train()  # Set model to training mode

        for epoch in range(EPOCHS):

            for batch_idx, batch in enumerate(batches):  # Looping over the batches
                current_states, optimal_actions, next_states = self.process_batch(batch)

                # Convert to tensors
                current_states_tensor = torch.FloatTensor(current_states).to(device)
                next_states_tensor = torch.FloatTensor(next_states).to(device)
                optimal_actions_tensor = torch.LongTensor(optimal_actions).to(device)

                # Prediction on S(t)
                estimated_qs_vec_t = self.model(current_states_tensor)
                estimated_qs_vec_t_np = estimated_qs_vec_t.detach().cpu().numpy()
                predicted_actions_t = self.greedy(
                    estimated_qs_vec_t_np, epsilon=EPSILON
                )  # Predict the actions based on epsilon-greedy algorithm
                rewards_t = self.get_reward(
                    predicted_actions_t, optimal_actions
                )  # Get the reward for each sample, the variable here unused because the rewarding phenomenon
                # is already done implicitly down while putting 1's in the q_values of optimal actions

                # Prediction on S(t+1)
                with torch.no_grad():
                    estimated_qs_vec_t_plus_one = self.model(next_states_tensor)
                    estimated_qs_vec_t_plus_one_np = (
                        estimated_qs_vec_t_plus_one.cpu().numpy()
                    )
                    predicted_actions_t_plus_one = self.greedy(
                        estimated_qs_vec_t_plus_one_np, epsilon=1.0
                    )  # Taking the always argmax (epsilon = 1.0)

                    # An np.arange object to access all rows, for vectorization
                    all_rows_idx = np.arange(estimated_qs_vec_t_plus_one_np.shape[0])

                    # Prediction with S(t+1) and a_cap(t+1)
                    q_cap_t_plus_one = estimated_qs_vec_t_plus_one_np[
                        all_rows_idx, predicted_actions_t_plus_one
                    ]  # Getting the q_values for the next predicted actions

                # Calculation of qref
                qref = np.zeros_like(
                    estimated_qs_vec_t_np
                )  # Set the qref shape and initialize as zeros
                qref[all_rows_idx, optimal_actions] = (
                    1  # Setting 1 to all values that correspond to the action of maximum value.
                )
                qref[all_rows_idx, predicted_actions_t] += (
                    LAMBDA * q_cap_t_plus_one
                )  # qref = rt + qcap_t+1
                qref_softmax = np.zeros_like(
                    qref
                )  # Softmax here is just for intuition, while what we do here is a hard max.
                qref_softmax[all_rows_idx, qref.argmax(1)] = (
                    1  # Replace the max value of the function by 1, all others by zeros. To act like classifier
                )

                # Convert target to tensor
                qref_softmax_tensor = torch.FloatTensor(qref_softmax).to(device)

                # Calculate loss and update
                self.optimizer.zero_grad()
                loss = F.cross_entropy(estimated_qs_vec_t, qref_softmax_tensor)
                loss.backward()

                # Gradient clipping (equivalent to clipnorm=1.0 in TensorFlow)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Calculate accuracy
                with torch.no_grad():
                    predictions = torch.argmax(estimated_qs_vec_t, dim=1)
                    targets = torch.argmax(qref_softmax_tensor, dim=1)
                    accuracy = (predictions == targets).float().mean().item()

                losses.append(loss.item())
                accuracies.append(accuracy)

                print(" -------------------------------------------------- ")
                print(
                    "In epoch {}/{} epochs, batch {}/{} batches:".format(
                        epoch + 1, EPOCHS, batch_idx + 1, self.data.n_train_batches
                    )
                )
                print("Accuracy: {}".format(accuracy))
                print("Loss: {}".format(loss.item()))
                print(" -------------------------------------------------- ")

        if save_path is not None:
            self.save_model(save_path)

        return losses, accuracies

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
                # Pad input if using CNN
                if self.use_cnn:
                    states = self.pad_input(states)
                states_tensor = torch.FloatTensor(states).to(device)
            else:
                states_tensor = states.to(device)

            estimated_qs = self.model(states_tensor)
            estimated_qs_np = estimated_qs.cpu().numpy()
            predicted_action = self.greedy(estimated_qs_np, epsilon=1.0).squeeze()
        return predicted_action


# ========================================================================================
# EXAMPLE USAGE AND DATA CREATION
# ========================================================================================

def create_sample_data(
    filename="sample_data.csv", n_samples=1000, n_features=64, n_actions=4
):
    """
    Creates a sample CSV file for testing the DQN
    Using 64 features (8x8 image) which works well with CNN
    """
    print(
        f"Creating sample data with {n_samples} samples, {n_features} features, {n_actions} actions..."
    )

    # Generate random state features
    np.random.seed(42)  # For reproducibility
    features = np.random.randn(n_samples, n_features)

    # Generate IDs
    ids = np.arange(n_samples)

    # Generate actions based on some simple logic
    feature_sums = np.sum(features, axis=1)
    actions = np.digitize(
        feature_sums,
        np.percentile(feature_sums, np.linspace(0, 100, n_actions + 1)[1:-1]),
    )
    actions = np.clip(actions, 0, n_actions - 1)

    # Create DataFrame
    columns = ["ID"] + [f"feature_{i}" for i in range(n_features)] + ["action"]
    data = np.column_stack([ids, features, actions])
    df = pd.DataFrame(data, columns=columns)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Sample data saved to {filename}")
    print(f"Data shape: {df.shape}")
    print(f"Action distribution: {np.bincount(actions.astype(int))}")
    return filename


def run_dqn_example():
    """
    Complete example of how to run the DQN training and testing
    """
    print("=" * 60)
    print("DEEP Q-NETWORK TRAINING EXAMPLE")
    print("=" * 60)

    # Step 1: Create or use existing data
    csv_file = "./data/dataset.csv"
    if not os.path.exists(csv_file):
        print("Creating sample data...")
        create_sample_data(csv_file, n_samples=2000, n_features=64, n_actions=3)  # 64 = 8x8 image
    else:
        print(f"Using existing data file: {csv_file}")

    print("\n" + "-" * 40)
    print("STEP 1: LOADING DATA")
    print("-" * 40)

    # Step 2: Load the data
    try:
        dataset = DataLoader(csv_file, batch_size=64)
        print(f"✓ Data loaded successfully!")
        print(f"  - Feature dimensions: {dataset.feature_dim}")
        print(f"  - Number of action classes: {dataset.actions_classes}")
        print(f"  - Training batches: {dataset.n_train_batches}")
        print(f"  - Testing batches: {dataset.n_test_batches}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    print("\n" + "-" * 40)
    print("STEP 2: CREATING DQN MODEL")
    print("-" * 40)

    # Step 3: Create the DQN model with CNN
    try:
        # Automatically infer image shape for CNN if possible
        feature_dim = dataset.feature_dim
        side_length = int(np.sqrt(feature_dim))
        if side_length * side_length == feature_dim:
            # Use CNN with correct image shape
            dqn = DeepQNet(dataset, use_cnn=True, img_height=side_length, img_width=side_length, img_channels=1)
            print(f"✓ DQN CNN model created successfully! (input: {side_length}x{side_length})")
        else:
            print(f"[WARNING] Feature count ({feature_dim}) is not a perfect square. Using FC model instead of CNN.")
            dqn = DeepQNet(dataset, use_cnn=False)
            print("✓ DQN FC model created successfully!")

        # Print model summary
        print("\nModel Architecture:")
        dqn.summary()

    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return

    print("\n" + "-" * 40)
    print("STEP 3: TRAINING THE MODEL")
    print("-" * 40)

    # Step 4: Train the model
    try:
        print("Starting training...")
        model_save_path = "trained_dqn_cnn_model.pth"
        losses, accuracies = dqn.train(save_path=model_save_path)

        print("✓ Training completed!")
        print(f"  - Final training loss: {losses[-1]:.4f}")
        print(f"  - Final training accuracy: {accuracies[-1]:.4f}")
        print(f"  - Model saved to: {model_save_path}")

    except Exception as e:
        print(f"✗ Error during training: {e}")
        return

    print("\n" + "-" * 40)
    print("STEP 4: TESTING THE MODEL")
    print("-" * 40)

    # Step 5: Test the model
    try:
        print("Starting testing...")
        test_accuracy, test_precision, test_recall, test_f1 = dqn.test()

        print("✓ Testing completed!")
        print(f"  - Test Results Summary:")
        print(f"    * Accuracy:  {test_accuracy:.4f}")
        print(f"    * Precision: {test_precision:.4f}")
        print(f"    * Recall:    {test_recall:.4f}")
        print(f"    * F1-Score:  {test_f1:.4f}")
        print(f"    * Model saved to: {model_save_path}")


    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return

    print("\n" + "-" * 40)
    print("STEP 5: MAKING PREDICTIONS")
    print("-" * 40)

    # Step 6: Make some sample predictions
    try:
        # Get some sample states from the test data
        sample_batch = dataset.test_batches[0][:5]  # First 5 samples from first test batch
        current_states, optimal_actions, _ = dqn.process_batch(sample_batch)

        # Make predictions
        predicted_actions = dqn.predict(current_states)

        print("Sample Predictions:")
        print("State Features | Optimal Action | Predicted Action | Match")
        print("-" * 60)
        for i in range(len(current_states)):
            features_str = f"[{', '.join([f'{x:.2f}' for x in current_states[i][:3]])}...]"  # Show first 3 features
            optimal = optimal_actions[i]
            predicted = (
                predicted_actions[i]
                if hasattr(predicted_actions, "__len__")
                else predicted_actions
            )
            match = "✓" if optimal == predicted else "✗"
            print(f"{features_str:20} | {optimal:13} | {predicted:15} | {match}")

    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        return

    print("\n" + "=" * 60)
    print("DQN CNN EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return dqn, dataset


if __name__ == "__main__":
    # Run the example
    print("here")
    dqn_model, data_loader = run_dqn_example()