import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional, Union
import json
from sklearn.metrics import confusion_matrix

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
        plt.style.use('default')  # Use default style instead of seaborn
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['grid.color'] = '#CCCCCC'
        plt.rcParams['axes.edgecolor'] = '#666666'
        plt.rcParams['axes.labelcolor'] = '#333333'
        plt.rcParams['xtick.color'] = '#333333'
        plt.rcParams['ytick.color'] = '#333333'
        
        # Set seaborn style if available
        try:
            sns.set_theme(style="whitegrid")
        except:
            pass  # Continue without seaborn if not available
    
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
    def __init__(self):
        self.metrics_data = {}
        self.setup_plot_style()
    
    def load_metrics(self, model_name: str) -> pd.DataFrame:
        """Load metrics from a model's log directory"""
        metrics_dir = f'logs/{model_name}_logs/metrics'
        if not os.path.exists(metrics_dir):
            print(f"Warning: No metrics directory found for {model_name}")
            return pd.DataFrame()
        
        # Get all CSV files in the metrics directory
        csv_files = [f for f in os.listdir(metrics_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"Warning: No CSV files found in {metrics_dir}")
            return pd.DataFrame()
        
        # Get the latest file
        latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(metrics_dir, x)))
        
        try:
            # Read CSV with error handling
            df = pd.read_csv(os.path.join(metrics_dir, latest_file), on_bad_lines='skip')
            self.metrics_data[model_name] = df
            return df
        except Exception as e:
            print(f"Warning: Error loading metrics for {model_name}: {str(e)}")
            return pd.DataFrame()
    
    def compare_models(self, metric: str, save_dir: str = 'comparison_results'):
        """Compare a specific metric across all models"""
        plt.figure(figsize=(12, 6))
        
        has_data_to_plot = False
        for model_name, df in self.metrics_data.items():
            if df.empty:
                continue

            for phase in ['train', 'val']:
                # Construct the column name based on phase and metric
                col_name = f"{phase}_{metric}"
                
                # Check if this column exists in the dataframe
                if col_name in df.columns:
                    # Get the series, drop NaN values to avoid plotting gaps
                    series = df[col_name].dropna()
                    if not series.empty:
                        # Plot the data
                        plt.plot(series.values, label=f'{model_name} ({phase.capitalize()})')
                        has_data_to_plot = True
        
        # Only save the plot if there is data
        if not has_data_to_plot:
            plt.close()  # Close the empty plot
            return None

        plt.title(f'{metric.replace("_", " ").title()} Comparison')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{metric}_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table of final metrics for all models"""
        summary_data = []
        
        for model_name, df in self.metrics_data.items():
            if df.empty:
                continue
                
            model_summary = {'model': model_name}
            
            # Get the latest metrics for each phase
            for phase in ['train', 'val', 'test']:
                phase_data = df[df['phase'] == phase] if 'phase' in df.columns else df
                if not phase_data.empty:
                    for col in df.columns:
                        if col not in ['phase', 'epoch', 'model']:
                            # Get the last value for this metric
                            last_value = phase_data[col].iloc[-1]
                            model_summary[f'{phase}_{col}'] = last_value
            
            summary_data.append(model_summary)
        
        return pd.DataFrame(summary_data)
    
    def setup_plot_style(self):
        """Setup consistent plot style for comparisons"""
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['grid.color'] = '#CCCCCC'
        plt.rcParams['axes.edgecolor'] = '#666666'
        plt.rcParams['axes.labelcolor'] = '#333333'
        plt.rcParams['xtick.color'] = '#333333'
        plt.rcParams['ytick.color'] = '#333333'
        
        try:
            sns.set_theme(style="whitegrid")
        except:
            pass 