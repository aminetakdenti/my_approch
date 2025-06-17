import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tracking_utils import ModelComparison

def compare_all_models():
    # Create timestamp for this comparison
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_dir = f'comparison_results_{timestamp}'
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Initialize model comparison
    model_comparison = ModelComparison()
    
    # List of models to compare
    models = [
        'dqn_nn',
        'double_dqn_nn',
        'dqn_cnn',
        'double_dqn_cnn'
    ]
    
    # Metrics to compare
    metrics = [
        'train_loss',
        'train_accuracy',
        'val_loss',
        'val_accuracy',
        'test_accuracy',
        'test_precision',
        'test_recall',
        'test_f1'
    ]
    
    # Load metrics for each model
    for model_name in models:
        model_comparison.load_metrics(model_name)
    
    # Generate comparison plots for each metric
    for metric in metrics:
        model_comparison.compare_models(
            metric=metric,
            save_dir=comparison_dir
        )
    
    # Create summary table
    summary_table = model_comparison.create_summary_table()
    summary_path = os.path.join(comparison_dir, 'model_comparison_summary.csv')
    summary_table.to_csv(summary_path, index=False)
    
    print(f"\nComparison results saved to: {comparison_dir}")
    print(f"Summary table saved to: {summary_path}")
    
    return comparison_dir

if __name__ == "__main__":
    compare_all_models() 