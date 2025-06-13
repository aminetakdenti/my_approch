import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the new project directory structure."""
    directories = [
        'src',
        'src/models',
        'src/data',
        'src/training',
        'src/utils',
        'src/config',
        'tests',
        'notebooks',
        'docs',
        'data/raw',
        'data/processed',
        'data/external',
        'models/cnn',
        'models/dqn',
        'models/hybrid',
        'logs/cnn',
        'logs/dqn',
        'logs/hybrid'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py files in Python packages
        if directory.startswith('src') or directory == 'tests':
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                open(init_file, 'a').close()

def move_files():
    """Move existing files to their new locations."""
    moves = [
        # Models
        ('cnn_models.py', 'src/models/cnn.py'),
        ('dqn_pytorch.py', 'src/models/dqn.py'),
        ('hybrid_dqn_agent.py', 'src/models/hybrid.py'),
        
        # Training scripts
        ('train_cnn.py', 'src/training/train_cnn.py'),
        ('DQN_RL_agent.py', 'src/training/train_dqn.py'),
        
        # Documentation
        ('cnn_tabular_paper.md', 'docs/cnn_tabular_paper.md'),
        ('cnn_models_explanation.md', 'docs/cnn_models_explanation.md'),
        ('DOCUMENTATION.md', 'docs/api_reference.md'),
        
        # Data
        ('data/dataset.csv', 'data/raw/dataset.csv'),
        
        # Models and logs
        ('trained_dqn_model.pth', 'models/dqn/trained_dqn_model.pth'),
        ('checkpoints/', 'models/cnn/checkpoints/'),
        ('logs/', 'logs/dqn/'),
    ]
    
    for src, dst in moves:
        if os.path.exists(src):
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            
            # Move the file/directory
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            print(f"Moved {src} to {dst}")

def create_config_file():
    """Create a basic configuration file."""
    config_content = """# Model configurations
models:
  cnn:
    input_channels: 1
    learning_rate: 0.001
    batch_size: 32
    epochs: 10
    
  dqn:
    learning_rate: 0.001
    gamma: 0.99
    epsilon_start: 1.0
    epsilon_end: 0.01
    
  hybrid:
    learning_rate: 0.001
    batch_size: 32

# Training configurations
training:
  early_stopping_patience: 3
  validation_split: 0.2
  save_dir: models

# Data configurations
data:
  raw_data_dir: data/raw
  processed_data_dir: data/processed
  external_data_dir: data/external

# Logging configurations
logging:
  log_dir: logs
  tensorboard: true
  wandb: false
"""
    
    with open('src/config/config.yaml', 'w') as f:
        f.write(config_content)

def main():
    print("Creating directory structure...")
    create_directory_structure()
    
    print("\nMoving files to new locations...")
    move_files()
    
    print("\nCreating configuration file...")
    create_config_file()
    
    print("\nProject restructuring complete!")
    print("Please review the changes and update any import statements in your code.")

if __name__ == "__main__":
    main() 