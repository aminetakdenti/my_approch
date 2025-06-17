import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional, Union
import json

class ModelTracker:
    def __init__(self, model_name: str, base_dir: str = "logs"):
        """
        Initialize the model tracker with organized directory structure.
        
        Args:
            model_name (str): Name of the model (e.g., 'dqn_cnn', 'double_dqn_cnn')
            base_dir (str): Base directory for all logs
        """
        self.model_name = model_name
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.dirs = {
            'plots': os.path.join(base_dir, f"{model_name}_logs", "plots"),
            'metrics': os.path.join(base_dir, f"{model_name}_logs", "metrics"),
            'checkpoints': os.path.join(base_dir, f"{model_name}_logs", "checkpoints"),
            'summaries': os.path.join(base_dir, f"{model_name}_logs", "summaries")
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epsilon': []
        }
        
        # Setup plot style
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Setup consistent plot style across all visualizations"""
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int, phase: str = 'train'):
        """
        Log metrics for a specific epoch and phase.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metrics to log
            epoch (int): Current epoch number
            phase (str): Phase of training ('train' or 'val')
        """
        # Add epoch and phase to metrics
        metrics['epoch'] = epoch
        metrics['phase'] = phase
        metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to CSV
        metrics_file = os.path.join(self.dirs['metrics'], f"{self.model_name}_metrics_{self.timestamp}.csv")
        df = pd.DataFrame([metrics])
        df.to_csv(metrics_file, mode='a', header=not os.path.exists(metrics_file), index=False)
        
        # Update metrics history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def plot_training_curves(self, save: bool = True):
        """
        Plot training curves for loss and accuracy.
        
        Args:
            save (bool): Whether to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.metrics_history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(self.metrics_history['val_loss'], label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.metrics_history['train_accuracy'], label='Train Accuracy', color='blue')
        ax2.plot(self.metrics_history['val_accuracy'], label='Val Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.dirs['plots'], f"{self.model_name}_training_curves_{self.timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_learning_rate_and_epsilon(self, save: bool = True):
        """
        Plot learning rate and epsilon curves.
        
        Args:
            save (bool): Whether to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot learning rate
        ax1.plot(self.metrics_history['learning_rate'], color='green')
        ax1.set_title('Learning Rate')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate')
        ax1.grid(True)
        
        # Plot epsilon
        ax2.plot(self.metrics_history['epsilon'], color='purple')
        ax2.set_title('Epsilon')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Epsilon')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.dirs['plots'], f"{self.model_name}_lr_epsilon_{self.timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            class_names: Optional[List[str]] = None, save: bool = True):
        """
        Plot confusion matrix.
        
        Args:
            y_true (List[int]): True labels
            y_pred (List[int]): Predicted labels
            class_names (Optional[List[str]]): Names of classes
            save (bool): Whether to save the plot
        """
        cm = pd.crosstab(pd.Series(y_true), pd.Series(y_pred))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save:
            save_path = os.path.join(self.dirs['plots'], f"{self.model_name}_confusion_matrix_{self.timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_summary(self, summary_dict: Dict[str, Union[float, str]]):
        """
        Save model summary to JSON file.
        
        Args:
            summary_dict (Dict[str, Union[float, str]]): Dictionary containing model summary
        """
        summary_file = os.path.join(self.dirs['summaries'], f"{self.model_name}_summary_{self.timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=4)

class ModelComparison:
    def __init__(self, base_dir: str = "logs"):
        """
        Initialize model comparison tool.
        
        Args:
            base_dir (str): Base directory containing model logs
        """
        self.base_dir = base_dir
    
    def load_metrics(self, model_name: str) -> pd.DataFrame:
        """
        Load metrics for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            pd.DataFrame: DataFrame containing model metrics
        """
        metrics_dir = os.path.join(self.base_dir, f"{model_name}_logs", "metrics")
        metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('.csv')]
        
        if not metrics_files:
            raise FileNotFoundError(f"No metrics files found for model {model_name}")
        
        # Load the most recent metrics file
        latest_file = max(metrics_files, key=lambda x: os.path.getctime(os.path.join(metrics_dir, x)))
        return pd.read_csv(os.path.join(metrics_dir, latest_file))
    
    def compare_models(self, model_names: List[str], metric: str = 'val_accuracy', 
                      save: bool = True, save_dir: str = "comparison_plots"):
        """
        Compare multiple models based on a specific metric.
        
        Args:
            model_names (List[str]): List of model names to compare
            metric (str): Metric to compare
            save (bool): Whether to save the plot
            save_dir (str): Directory to save comparison plots
        """
        plt.figure(figsize=(12, 6))
        
        for model_name in model_names:
            df = self.load_metrics(model_name)
            plt.plot(df['epoch'], df[metric], label=model_name)
        
        plt.title(f'Comparison of {metric} across models')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        
        if save:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"comparison_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_summary_table(self, model_names: List[str], metrics: List[str]) -> pd.DataFrame:
        """
        Create a summary table comparing multiple models across different metrics.
        
        Args:
            model_names (List[str]): List of model names to compare
            metrics (List[str]): List of metrics to compare
            
        Returns:
            pd.DataFrame: Summary table
        """
        summary_data = []
        
        for model_name in model_names:
            df = self.load_metrics(model_name)
            model_summary = {'model': model_name}
            
            for metric in metrics:
                model_summary[f'{metric}_mean'] = df[metric].mean()
                model_summary[f'{metric}_max'] = df[metric].max()
                model_summary[f'{metric}_min'] = df[metric].min()
            
            summary_data.append(model_summary)
        
        return pd.DataFrame(summary_data) 