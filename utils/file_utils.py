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

def create_training_directory(base_dir: str ='./models', training_subdir: str ='trainings') -> tuple:
    """
    Creates a directory structure for training, including subdirectories for checkpoints and logs.
    
    Parameters:
    base_dir (str): The base directory where the training directories should be created. Default is './models'.
    training_subdir (str): The subdirectory under the base directory for training sessions. Default is 'trainings'.

    Usage:
    training_dir, checkpoints_dir, logs_dir = create_training_directory(base_dir='./models', training_subdir='trainings')

    Returns:
    tuple: A tuple containing paths to the training directory, checkpoints directory, and logs directory.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    training_dir = os.path.join(base_dir, training_subdir, timestamp)
    checkpoints_dir = os.path.join(training_dir, 'checkpoints')
    logs_dir = os.path.join(training_dir, 'logs')

    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    return training_dir, checkpoints_dir, logs_dir

def create_evaluation_directory(base_dir: str ='./models', evaluation_subdir: str ='evaluations') -> str:
    """
    Creates a directory structure for evaluation.

    Parameters:
    base_dir (str): The base directory where the evaluation directories should be created. Default is './models'.
    evaluation_subdir (str): The subdirectory under the base directory for evaluation sessions. Default is 'evaluations'.

    Usage:
    evaluation_dir = create_evaluation_directory(base_dir='./models', evaluation_subdir='evaluations')

    Returns:
    str: The path to the created evaluation directory.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    evaluation_dir = os.path.join(base_dir, evaluation_subdir, timestamp)

    os.makedirs(evaluation_dir, exist_ok=True)

    return evaluation_dir

def load_config(config_path: str='config.yaml') -> dict:
    """
    Loads a YAML configuration file.

    Parameters:
    config_path (str): The path to the configuration file. Default is 'config.yaml'.

    Usage:
    config = load_config(config_path='configs/cds.yaml')

    Returns:
    dict: The configuration settings loaded from the YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config