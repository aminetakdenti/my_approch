import argparse
from tracking_utils import ModelComparison
import pandas as pd
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Compare different model results')
    parser.add_argument('--models', nargs='+', required=True, help='List of model names to compare')
    parser.add_argument('--metrics', nargs='+', default=['val_accuracy', 'val_loss'], 
                       help='List of metrics to compare')
    parser.add_argument('--save-dir', default='comparison_results', 
                       help='Directory to save comparison results')
    args = parser.parse_args()
    
    # Create comparison tool
    comparator = ModelComparison()
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Compare each metric
    for metric in args.metrics:
        print(f"\nComparing {metric} across models...")
        comparator.compare_models(
            model_names=args.models,
            metric=metric,
            save=True,
            save_dir=save_dir
        )
    
    # Create summary table
    print("\nCreating summary table...")
    summary_table = comparator.create_summary_table(args.models, args.metrics)
    
    # Save summary table
    summary_path = os.path.join(save_dir, 'model_comparison_summary.csv')
    summary_table.to_csv(summary_path, index=False)
    print(f"Summary table saved to: {summary_path}")
    
    # Print summary table
    print("\nModel Comparison Summary:")
    print(summary_table.to_string(index=False))

if __name__ == "__main__":
    main() 