import os
import torch
import numpy as np
from double_dqn_nn import DoubleDeepQNet, DataLoader
from model_utils import ModelLogger, ModelVisualizer
from typing import Dict, Any
import json

def main(
    data_path: str = 'data/processed_data.csv',
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 0.001,
    model_name: str = 'double_dqn_model',
    save_dir: str = 'models',
    log_dir: str = 'logs'
) -> Dict[str, Any]:
    """
    Main function to train and evaluate a Double DQN model.
    
    Args:
        data_path (str): Path to the processed data CSV file
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer
        model_name (str): Name for saving the model
        save_dir (str): Directory to save the model
        log_dir (str): Directory to save logs and visualizations
    
    Returns:
        Dict[str, Any]: Dictionary containing training results and metrics
    """
    try:
        # Create necessary directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data_loader = DataLoader(data_path, batch_size=batch_size)
        
        # Initialize model
        print("Initializing Double DQN model...")
        model = DoubleDeepQNet(data_loader)
        model.summary()
        
        # Train model
        print("Starting training...")
        model_path = os.path.join(save_dir, f"{model_name}.pth")
        losses, accuracies = model.train(save_path=model_path, plot_dir=log_dir)
        
        # Evaluate model
        print("Evaluating model...")
        accuracy, precision, recall, f1 = model.test()
        
        # Save results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'final_loss': losses[-1],
            'final_accuracy': accuracies[-1]
        }
        
        # Save results to JSON
        results_path = os.path.join(log_dir, f"{model_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print("\nTraining Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"\nResults saved to: {results_path}")
        print(f"Model saved to: {model_path}")
        
        return results
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Get parameters from environment variables or use defaults
    data_path = os.getenv('DATA_PATH', 'data/dataset.csv')
    batch_size = int(os.getenv('BATCH_SIZE', '32'))
    epochs = int(os.getenv('EPOCHS', '100'))
    learning_rate = float(os.getenv('LEARNING_RATE', '0.001'))
    model_name = os.getenv('MODEL_NAME', 'double_dqn_model')
    save_dir = os.getenv('SAVE_DIR', 'models')
    log_dir = os.getenv('LOG_DIR', 'logs')
    
    # Run training
    main(
        data_path=data_path,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        model_name=model_name,
        save_dir=save_dir,
        log_dir=log_dir
    )