import glob
import os
import yaml
from datetime import datetime

def get_file_paths(dir: str) -> list:
    """
    Retrieves a list of files matching the specified directory pattern.
    
    Parameters:
    dir (str): The directory pattern to search for files.

    Usage:
    file_paths = get_file_paths('data/*.nc')

    Returns:
    list: A list of file paths that match the specified directory pattern.
    """
    files = glob.glob(dir)
    print(f"[INFO] Found {len(files)} files in {dir}")
    return files

def create_training_directory(base_dir='./model/trainings'):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    training_dir = os.path.join(base_dir, timestamp)
    checkpoints_dir = os.path.join(training_dir, 'checkpoints')
    logs_dir = os.path.join(training_dir, 'logs')

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    return training_dir, checkpoints_dir, logs_dir

def create_evaluation_directory(base_dir='./evaluation/evaluations'):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    evaluation_dir = os.path.join(base_dir, timestamp)

    os.makedirs(evaluation_dir, exist_ok=True)

    return evaluation_dir

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config