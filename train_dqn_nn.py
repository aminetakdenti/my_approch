import pandas as pd
import os

# [Include all the DQN code from previous artifact here - I'll assume it's in a separate file]
from dqn_nn import DeepQNet, DataLoader

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
    dqn_model, data_loader = run_with_custom_data("data/dataset.csv")


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
