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

def create_training_directory(log_config: dict) -> tuple:
    """
    Creates a directory structure for training, including subdirectories for checkpoints and logs.
    
    Parameters:
    log_config (dict): A dictionary containing configuration for log directories.

    Usage:
    training_dir, checkpoint_dir, log_dir, inference_dir = create_training_directory(config['logs'])

    Returns:
    tuple: A tuple containing paths to the training directory, checkpoints directory, logs directory, and inference result.
    """
    base_dir = log_config['base_dir']
    training_subdir = log_config['training_subdir']
    inference_subdir = log_config['inference_subdir']
    checkpoint_subdir = log_config['checkpoint_subdir']
    log_subdir = log_config['log_subdir']

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Training Logs
    training_dir = os.path.join(base_dir, training_subdir, timestamp)
    checkpoint_dir = os.path.join(training_dir, checkpoint_subdir)
    log_dir = os.path.join(training_dir, log_subdir)
    inference_dir = os.path.join(training_dir, inference_subdir)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(inference_dir, exist_ok=True)

    return training_dir, checkpoint_dir, log_dir, inference_dir

def create_evaluation_directory(log_config: dict) -> tuple:
    """
    Creates a directory structure for evaluation.

    Parameters:
    log_config (dict): A dictionary containing configuration for log directories.

    Usage:
    evaluation_dir, log_dir, inference_dir = create_evaluation_directory(config['logs'])

    Returns:
    tuple: A tuple containing paths to the evaluation directory, logs directory, and inference result.
    """
    base_dir = log_config['base_dir']
    evaluation_subdir = log_config['evaluation_subdir']
    inference_subdir = log_config['inference_subdir']
    log_subdir = log_config['log_subdir']

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Evaluation Logs
    evaluation_dir = os.path.join(base_dir, evaluation_subdir, timestamp)
    log_dir = os.path.join(evaluation_dir, log_subdir)
    inference_dir = os.path.join(evaluation_dir, inference_subdir)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(inference_dir, exist_ok=True)

    return evaluation_dir, log_dir, inference_dir

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

def dump_config(config: dict, path: str) -> None:
    """
    Dumps the given configuration dictionary into a YAML file at the specified path.

    Parameters:
    config (Dict): The configuration dictionary to be saved.
    path (str): The file path where the YAML file will be saved.

    Returns:
    None
    """
    with open(path, 'w') as file:
        yaml.dump(config, file)