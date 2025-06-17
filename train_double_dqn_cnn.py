import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from double_dqn_cnn import DoubleCNNModel, DoubleCNNTrainer

class CSVDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def prepare_data(csv_path, test_size=0.2, batch_size=32):
    # Read the CSV file
    print("Loading data from CSV...")
    df = pd.read_csv(csv_path)
    
    # Separate features and labels
    # Assuming the last column is the label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Calculate the dimensions for reshaping
    n_features = X.shape[1]
    # Find the closest square root for the number of features
    side_length = int(np.ceil(np.sqrt(n_features)))
    # Pad the features to make it a square
    X_train_padded = np.pad(X_train, ((0, 0), (0, side_length**2 - n_features)), 'constant')
    X_test_padded = np.pad(X_test, ((0, 0), (0, side_length**2 - n_features)), 'constant')
    
    # Reshape the data to 2D images
    X_train_reshaped = X_train_padded.reshape(-1, 1, side_length, side_length)
    X_test_reshaped = X_test_padded.reshape(-1, 1, side_length, side_length)
    
    # Create datasets
    train_dataset = CSVDataset(X_train_reshaped, y_train)
    test_dataset = CSVDataset(X_test_reshaped, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, side_length

def main():
    # Parameters
    csv_path = 'data/dataset.csv'
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    
    # Prepare data
    print("Preparing data...")
    train_loader, test_loader, image_size = prepare_data(csv_path, batch_size=batch_size)
    
    # Get number of classes from the data
    num_classes = len(np.unique(pd.read_csv(csv_path).iloc[:, -1]))
    
    # Create model
    print(f"Creating Double DQN CNN model with input size {image_size}x{image_size} and {num_classes} classes...")
    model = DoubleCNNModel(
        input_channels=1,
        height=image_size,
        width=image_size,
        output_dim=num_classes
    )
    
    # Create trainer
    trainer = DoubleCNNTrainer(model, learning_rate=learning_rate)
    
    # Train the model
    print("Starting training...")
    losses, accuracies, best_acc = trainer.run_training(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        save_dir='checkpoints',
        early_stopping_patience=3
    )
    
    print(f"\nTraining completed! Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main() 