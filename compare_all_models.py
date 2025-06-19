import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tracking_utils import ModelComparison

def compare_all_models():
    """Compare all trained models and generate comparison plots"""
    # Create timestamped directory for results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'logs/comparison_results_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model comparison
    model_comparison = ModelComparison()
    
    # List of models to compare
    model_names = [
        'dqn_nn',
        'double_dqn_nn',
        'dqn_cnn',
        'double_dqn_cnn'
    ]
    
    # Load metrics for each model
    for model_name in model_names:
        print(f"Loading metrics for {model_name}...")
        model_comparison.load_metrics(model_name)
    
    # Metrics to compare
    metrics = [
        'loss',
        'accuracy',
        'learning_rate',
        'epsilon'
    ]
    
    # Generate comparison plots for each metric
    for metric in metrics:
        print(f"Generating comparison plot for {metric}...")
        plot_path = model_comparison.compare_models(metric, save_dir)
        print(f"Saved plot to: {plot_path}")
    
    # Create and save summary table
    print("Generating summary table...")
    summary_df = model_comparison.create_summary_table()
    summary_path = os.path.join(save_dir, 'model_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table to: {summary_path}")
    
    print("\nComparison completed! Results saved in:", save_dir)

if __name__ == "__main__":
    compare_all_models() 