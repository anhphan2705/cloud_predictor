import argparse
import os
import sys
import torch
from tools.data_process import data_pipeline
from tools.train import train_pipeline
from tools.eval import evaluate_pipeline
from utils.file_utils import create_training_directory, create_evaluation_directory, load_config, dump_config

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()

    def close(self):
        self.log.close()

def main(config: dict) -> None:
    """
    Main function to run the weather forecasting with Temporal Fusion Transformer.
    
    Depending on the mode ('train' or 'eval'), it will create necessary directories, load data, train the model, 
    or evaluate the model.

    Parameters:
    config (dict): Configuration dictionary loaded from a YAML file.
    """
    data_config = config['data']
    time_series_config = config['time_series']
    training_config = config['training']
    evaluation_config = config['evaluation']
    log_config = config['logging']

    # Create directories for logs and checkpoints
    if args.mode == 'train':
        training_dir, checkpoint_dir, logs_dir, inference_dir = create_training_directory(log_config)

        # Dump the configuration file
        dump_config(config, os.path.join(logs_dir, "config.yaml"))

        # Redirect stdout and stderr to log file
        sys.stdout = Logger(os.path.join(logs_dir, "training_log.txt"))
        sys.stderr = sys.stdout

        try:
            # Load the data
            train_dataloader, val_dataloader = data_pipeline(
                data_root=data_config['data_root'],
                data_source=data_config['data_source'],
                time_series_config=time_series_config,
                time_column=data_config['time_column'],
                batch_size=training_config['batch_size'],
                num_workers=training_config['num_workers'],
                save_dir=data_config['save_dir'],
                mode='train'
            )

            # Train the model
            train_pipeline(train_dataloader, val_dataloader, training_dir, checkpoint_dir, logs_dir, inference_dir, config)

        finally:
            # Restore the original stdout and stderr
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    elif args.mode == 'eval':
        evaluation_dir, logs_dir, inference_dir = create_evaluation_directory(log_config)

        # Dump the configuration file
        dump_config(config, os.path.join(logs_dir, "config.yaml"))

        # Redirect stdout and stderr to log file
        sys.stdout = Logger(os.path.join(logs_dir, "evaluation_log.txt"))
        sys.stderr = sys.stdout

        try:
            # Load the inference data
            _, eval_dataloader = data_pipeline(
                data_root=evaluation_config['data_root'],
                data_source=data_config['data_source'],
                time_series_config=time_series_config,
                time_column=data_config['time_column'],
                batch_size=training_config['batch_size'],
                num_workers=training_config['num_workers'],
                save_dir=data_config['save_dir'],
                mode='eval'
            )
            
            model_path = evaluation_config['model_path']

            # Evaluate the model
            evaluate_pipeline(model_path, eval_dataloader, inference_dir)

        finally:
            # Restore the original stdout and stderr
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weather Forecasting with Temporal Fusion Transformer')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, help='Mode to run: train or eval')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file (REQUIRED)')
    parser.add_argument('--cuda_memory_fraction', type=float, default=0.5, help='Fraction of CUDA memory to use (e.g., 0.5 for 50%)')
    args = parser.parse_args()

    if args.cuda_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)

    config = load_config(args.config)
    main(config)