import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score
import os

# [Include all the DQN code from previous artifact here - I'll assume it's in a separate file]
from dqn_pytorch import DeepQNet, DataLoader

# ========================================================================================
# STEP 1: CREATE SAMPLE DATA (if you don't have your own CSV file)
# ========================================================================================


def create_sample_data(
    filename="sample_data.csv", n_samples=1000, n_features=10, n_actions=4
):
    """
    Creates a sample CSV file for testing the DQN

    Args:
        filename: Name of the CSV file to create
        n_samples: Number of data samples
        n_features: Number of state features
        n_actions: Number of possible actions (0, 1, 2, ..., n_actions-1)
    """
    print(
        f"Creating sample data with {n_samples} samples, {n_features} features, {n_actions} actions..."
    )

    # Generate random state features
    np.random.seed(42)  # For reproducibility
    features = np.random.randn(n_samples, n_features)

    # Generate IDs
    ids = np.arange(n_samples)

    # Generate actions based on some simple logic (you can modify this)
    # Example: action depends on sum of features
    feature_sums = np.sum(features, axis=1)
    actions = np.digitize(
        feature_sums,
        np.percentile(feature_sums, np.linspace(0, 100, n_actions + 1)[1:-1]),
    )
    actions = np.clip(actions, 0, n_actions - 1)  # Ensure actions are in valid range

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


# ========================================================================================
# STEP 2: MAIN EXECUTION FUNCTION
# ========================================================================================


def run_dqn_example():
    """
    Complete example of how to run the DQN training and testing
    """
    print("=" * 60)
    print("DEEP Q-NETWORK TRAINING EXAMPLE")
    print("=" * 60)

    # Step 1: Create or use existing data
    csv_file = "sample_data.csv"
    if not os.path.exists(csv_file):
        print("Creating sample data...")
        create_sample_data(csv_file, n_samples=2000, n_features=8, n_actions=3)
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

    # Step 3: Create the DQN model
    try:
        dqn = DeepQNet(dataset)
        print("✓ DQN model created successfully!")

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
        model_save_path = "trained_dqn_model.pth"
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

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return

    print("\n" + "-" * 40)
    print("STEP 5: MAKING PREDICTIONS")
    print("-" * 40)

    # Step 6: Make some sample predictions
    try:
        # Get some sample states from the test data
        sample_batch = dataset.test_batches[0][
            :5
        ]  # First 5 samples from first test batch
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
    print("DQN EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return dqn, dataset


# ========================================================================================
# STEP 3: HOW TO RUN WITH YOUR OWN DATA
# ========================================================================================


def run_with_custom_data(csv_filepath):
    """
    Example of how to run with your own CSV data

    Your CSV should have the format:
    ID, feature1, feature2, ..., featureN, action

    Where:
    - ID: Unique identifier for each sample
    - feature1, feature2, etc.: State features (numerical values)
    - action: The optimal action (integer: 0, 1, 2, ...)
    """
    print(f"Running DQN with custom data: {csv_filepath}")

    # Check if file exists
    if not os.path.exists(csv_filepath):
        print(f"Error: File {csv_filepath} not found!")
        return

    # Load and inspect the data first
    df = pd.read_csv(csv_filepath)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())

    # Load with DataLoader
    dataset = DataLoader(csv_filepath, batch_size=64)

    # Create and train model
    dqn = DeepQNet(dataset)
    losses, accuracies = dqn.train(save_path="custom_model.pth")

    # Test the model
    test_results = dqn.test()

    return dqn, dataset


# ========================================================================================
# MAIN EXECUTION
# ========================================================================================

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run with sample data (automatically generated)")
    print("2. Run with your own CSV file")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        # Run with sample data
        dqn_model, data_loader = run_dqn_example()

    elif choice == "2":
        # Run with custom data
        csv_path = input("Enter path to your CSV file: ").strip()
        dqn_model, data_loader = run_with_custom_data(csv_path)

    else:
        print("Invalid choice. Running with sample data...")
        dqn_model, data_loader = run_dqn_example()

# ========================================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ========================================================================================


def load_trained_model(model_path, dataset):
    """
    Load a previously trained model
    """
    dqn = DeepQNet(dataset)
    dqn.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return dqn


def plot_training_history(losses, accuracies):
    """
    Plot training loss and accuracy (requires matplotlib)
    """
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(losses)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Batch")
        ax1.set_ylabel("Loss")

        ax2.plot(accuracies)
        ax2.set_title("Training Accuracy")
        ax2.set_xlabel("Batch")
        ax2.set_ylabel("Accuracy")

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("matplotlib not available for plotting")
