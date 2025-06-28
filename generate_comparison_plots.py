import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from tracking_utils import ModelComparison

def generate_comparison_plots():
    """
    Load metrics for all models, identify all available metrics,
    and generate comparison plots and a summary table.
    """
    # Create timestamped directory for results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'logs/thesis_comparison_{timestamp}'
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
    
    # Load summary data from JSON files
    summary_data = {}
    for model_name in model_names:
        summary_file = f'logs/{model_name}_logs/summaries/{model_name}_summary_*.json'
        import glob
        files = glob.glob(summary_file)
        if files:
            # Get the most recent summary file
            latest_file = max(files, key=os.path.getctime)
            try:
                with open(latest_file, 'r') as f:
                    summary_data[model_name] = json.load(f)
                print(f"Loaded summary for {model_name}: {latest_file}")
            except Exception as e:
                print(f"Error loading summary for {model_name}: {e}")
                summary_data[model_name] = {}
        else:
            print(f"No summary file found for {model_name}")
            summary_data[model_name] = {}
    
    # Create comprehensive comparison table
    comparison_data = []
    for model_name, data in summary_data.items():
        if data:
            row = {
                'Model': model_name,
                'Accuracy': data.get('test_accuracy', 'N/A'),
                'Precision': data.get('test_precision', 'N/A'),
                'Recall': data.get('test_recall', 'N/A'),
                'F1_Score': data.get('test_f1', 'N/A'),
                'Learning_Rate': data.get('LEARNING_RATE', 'N/A'),
                'Epochs': data.get('EPOCHS', 'N/A'),
                'Batch_Size': data.get('BATCH_SIZE', 'N/A'),
                'Total_Params': data.get('total_params', 'N/A'),
                'Input_Dim': data.get('input_dim', data.get('input_channels', 'N/A')),
                'Output_Dim': data.get('output_dim', 'N/A'),
                'Epsilon_Start': data.get('EPSILON_START', 'N/A'),
                'Epsilon_End': data.get('EPSILON_END', 'N/A'),
                'Epsilon_Decay': data.get('EPSILON_DECAY', 'N/A'),
                'Lambda': data.get('LAMBDA', 'N/A')
            }
            comparison_data.append(row)
    
    # Create and save comparison table
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(f'{save_dir}/comprehensive_model_comparison.csv', index=False)
        
        # Print the table
        print("\n" + "="*100)
        print("COMPREHENSIVE MODEL COMPARISON TABLE")
        print("="*100)
        print(comparison_df.to_string(index=False))
        print("="*100)
        
        # Create comparison plots for key metrics
        create_comparison_plots(comparison_df, save_dir)
    
    # Also generate the original comparison plots from training metrics
    print("\nGenerating training metrics comparison plots...")
    for metric in ['accuracy', 'loss']:
        try:
            model_comparison.compare_models(metric, save_dir)
            print(f"Generated {metric} comparison plot")
        except Exception as e:
            print(f"Error generating {metric} plot: {e}")

def create_comparison_plots(df, save_dir):
    """Create comparison plots for the key metrics"""
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Filter out 'N/A' values for plotting
    plot_df = df.copy()
    numeric_columns = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    
    for col in numeric_columns:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    
    # 1. Performance Metrics Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    x = np.arange(len(df['Model']))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = plot_df[metric].values
        ax.bar(x + i*width, values, width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison (Accuracy, Precision, Recall, F1)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df['Model'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. F1 Score Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['Model'], plot_df['F1_Score'], color='skyblue', alpha=0.8)
    ax.set_xlabel('Models')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison Across Models')
    ax.set_xticklabels(df['Model'], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, plot_df['F1_Score']):
        if not pd.isna(value):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                   f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/f1_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['Model'], plot_df['Accuracy'], color='lightgreen', alpha=0.8)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison Across Models')
    ax.set_xticklabels(df['Model'], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, plot_df['Accuracy']):
        if not pd.isna(value):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                   f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Model Complexity Comparison (Parameters)
    fig, ax = plt.subplots(figsize=(10, 6))
    param_counts = pd.to_numeric(df['Total_Params'], errors='coerce')
    bars = ax.bar(df['Model'], param_counts, color='orange', alpha=0.8)
    ax.set_xlabel('Models')
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Model Complexity Comparison (Total Parameters)')
    ax.set_xticklabels(df['Model'], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, param_counts):
        if not pd.isna(value):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_counts)*0.01, 
                   f'{int(value):,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_complexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated comparison plots in {save_dir}")

if __name__ == "__main__":
    import numpy as np
    generate_comparison_plots()
    print("\nComparison generation completed!") 