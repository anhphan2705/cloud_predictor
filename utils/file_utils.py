import glob
import os
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

def create_training_directory(base_path='./model'):
    # Get current date and time
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    # Create directory path
    training_dir = os.path.join(base_path, current_time)
    checkpoints_dir = os.path.join(training_dir, 'checkpoints')
    logs_dir = os.path.join(training_dir, 'logs')

    # Create directories
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    return training_dir, checkpoints_dir, logs_dir