import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import torch
import pandas as pd
from functools import wraps

def csv_logger(save_dir='logs', prefix=''):
    """
    Decorator to automatically save metrics to CSV while preserving original method functionality
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the original function
            result = func(*args, **kwargs)
            
            # Extract metrics based on the function name
            metrics = {}
            if func.__name__ == 'log_model_summary':
                model = args[0]
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                metrics = {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_architecture': str(model)
                }
                prefix_suffix = 'model_summary_'
                
            elif func.__name__ == 'log_training_start':
                epochs, device, model_name = args[0], args[1], args[2] if len(args) > 2 else ""
                metrics = {
                    'model_name': model_name,
                    'epochs': epochs,
                    'device': str(device),
                    'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                prefix_suffix = 'training_start_'
                
            elif func.__name__ == 'log_training_end':
                best_accuracy = args[0]
                final_metrics = args[1] if len(args) > 1 else None
                metrics = {
                    'best_accuracy': best_accuracy,
                    'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                if final_metrics:
                    metrics.update(final_metrics)
                prefix_suffix = 'training_end_'
            
            # Save metrics to CSV
            if metrics:
                ModelLogger.save_metrics_to_csv(
                    metrics,
                    save_dir=save_dir,
                    prefix=f'{prefix}{prefix_suffix}'
                )
            
            return result
        return wrapper
    return decorator

class ModelVisualizer:
    @staticmethod
    def setup_plot_style():
        """Setup consistent plot style across all visualizations"""
        plt.style.use('bmh')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

    @staticmethod
    def plot_training_metrics(history, save_dir='logs', prefix=''):
        """
        Plot training metrics including loss, accuracy, and other metrics.
        
        Args:
            history (dict): Dictionary containing training metrics
            save_dir (str): Directory to save the plots
            prefix (str): Prefix for the saved files
        """
        ModelVisualizer.setup_plot_style()
        
        # Create figure with subplots
        n_metrics = len(history)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        # Plot each metric
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        for idx, (metric_name, values) in enumerate(history.items()):
            ax = axes[idx]
            
            # Plot raw values
            ax.plot(values, label=f'{metric_name}', color=colors[idx % len(colors)], alpha=0.3, linewidth=1)
            
            # Add moving average
            window_size = min(50, len(values))
            if window_size > 1:
                values_smooth = np.convolve(np.array(values), np.ones(window_size)/window_size, mode='valid')
                ax.plot(np.arange(window_size-1, len(values)), values_smooth, 
                       label=f'{metric_name} (MA)', color=colors[idx % len(colors)], linewidth=2)
            
            ax.set_title(f'{metric_name} Over Time')
            ax.set_xlabel('Batch')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(history), len(axes)):
            axes[idx].set_visible(False)
        
        # Save plot
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'{prefix}training_metrics_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, save_dir='logs', prefix=''):
        """Plot confusion matrix"""
        ModelVisualizer.setup_plot_style()
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        
        # Add value annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'{prefix}confusion_matrix_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

    @staticmethod
    def plot_class_metrics(y_true, y_pred, classes, save_dir='logs', prefix=''):
        """Plot precision, recall, and F1-score for each class"""
        ModelVisualizer.setup_plot_style()
        
        # Calculate metrics for each class
        precisions = precision_score(y_true, y_pred, labels=range(len(classes)), average=None, zero_division=0)
        recalls = recall_score(y_true, y_pred, labels=range(len(classes)), average=None, zero_division=0)
        f1_scores = f1_score(y_true, y_pred, labels=range(len(classes)), average=None, zero_division=0)
        
        # Create bar plot
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precisions, width, label='Precision', color='skyblue')
        ax.bar(x, recalls, width, label='Recall', color='lightgreen')
        ax.bar(x + width, f1_scores, width, label='F1-score', color='salmon')
        
        # Customize plot
        ax.set_ylabel('Score')
        ax.set_title('Class-wise Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
        
        for container in ax.containers:
            add_value_labels(container)
        
        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'{prefix}class_metrics_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path, {
            'precision': precisions,
            'recall': recalls,
            'f1': f1_scores
        }

class ModelLogger:
    @staticmethod
    def log_metrics(metrics, epoch, batch_idx=None, total_batches=None):
        """Log training metrics with consistent formatting"""
        if batch_idx is not None and total_batches is not None:
            print(f"Epoch [{epoch+1}/{metrics.get('epochs', '?')}] "
                  f"Batch [{batch_idx+1}/{total_batches}]")
        else:
            print(f"Epoch [{epoch+1}/{metrics.get('epochs', '?')}]")
            
        for metric_name, value in metrics.items():
            if metric_name != 'epochs':
                if isinstance(value, float):
                    print(f"{metric_name}: {value:.4f}")
                else:
                    print(f"{metric_name}: {value}")
        print("--------------------------------------------------")

    @staticmethod
    @csv_logger()
    def log_model_summary(model):
        """Log model architecture and parameters"""
        print("\nModel Summary:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("--------------------------------------------------")

    @staticmethod
    @csv_logger()
    def log_training_start(epochs, device, model_name=""):
        """Log training start information"""
        print(f"\nStarting training for {model_name}")
        print(f"Training for {epochs} epochs")
        print(f"Using device: {device}")
        print("--------------------------------------------------")

    @staticmethod
    @csv_logger()
    def log_training_end(best_accuracy, final_metrics=None):
        """Log training end information"""
        print("\nTraining completed!")
        print(f"Best accuracy: {best_accuracy:.4f}")
        if final_metrics:
            print("\nFinal metrics:")
            for metric_name, value in final_metrics.items():
                print(f"{metric_name}: {value:.4f}")
        print("--------------------------------------------------")

    @staticmethod
    def save_metrics_to_csv(metrics_dict, y_true=None, y_pred=None, classes=None, save_dir='logs', prefix=''):
        """
        Save various metrics to a CSV file including confusion matrix, F1, recall, accuracy etc.
        
        Args:
            metrics_dict (dict): Dictionary containing training metrics
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            classes (list): List of class names
            save_dir (str): Directory to save the CSV file
            prefix (str): Prefix for the saved file
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create a dictionary to store all metrics
        all_metrics = {}
        
        # Add basic metrics
        all_metrics.update(metrics_dict)
        
        # Add classification metrics if y_true and y_pred are provided
        if y_true is not None and y_pred is not None:
            all_metrics['accuracy'] = accuracy_score(y_true, y_pred)
            all_metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            all_metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            all_metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            
            # Add per-class metrics if classes are provided
            if classes is not None:
                for i, class_name in enumerate(classes):
                    class_f1 = f1_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
                    class_precision = precision_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
                    class_recall = recall_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
                    
                    all_metrics[f'{class_name}_f1'] = class_f1
                    all_metrics[f'{class_name}_precision'] = class_precision
                    all_metrics[f'{class_name}_recall'] = class_recall
            
            # Add confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            if classes is not None:
                for i, true_class in enumerate(classes):
                    for j, pred_class in enumerate(classes):
                        all_metrics[f'confusion_{true_class}_vs_{pred_class}'] = cm[i, j]
            else:
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        all_metrics[f'confusion_{i}_vs_{j}'] = cm[i, j]
        
        # Convert to DataFrame and save
        df = pd.DataFrame([all_metrics])
        save_path = os.path.join(save_dir, f'{prefix}metrics_{timestamp}.csv')
        df.to_csv(save_path, index=False)
        
        return save_path 