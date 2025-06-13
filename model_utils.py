import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import torch

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
    def log_training_start(epochs, device, model_name=""):
        """Log training start information"""
        print(f"\nStarting training for {model_name}")
        print(f"Training for {epochs} epochs")
        print(f"Using device: {device}")
        print("--------------------------------------------------")

    @staticmethod
    def log_training_end(best_accuracy, final_metrics=None):
        """Log training end information"""
        print("\nTraining completed!")
        print(f"Best accuracy: {best_accuracy:.4f}")
        if final_metrics:
            print("\nFinal metrics:")
            for metric_name, value in final_metrics.items():
                print(f"{metric_name}: {value:.4f}")
        print("--------------------------------------------------") 