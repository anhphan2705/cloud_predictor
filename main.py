import argparse
import os
import torch
from tools.data_process import data_pipeline
from tools.train import train_pipeline
from tools.eval import evaluate_model
from utils.file_utils import create_training_directory, create_evaluation_directory, load_config

def main(config):
    data_config = config['data']
    training_config = config['training']
    evaluation_config = config['evaluation']
    checkpoints_config = config['checkpoints']

    # Create directories for logs and checkpoints
    if args.mode == 'train':
        training_dir, checkpoints_dir, logs_dir = create_training_directory(base_dir=checkpoints_config['base_dir'], training_subdir=checkpoints_config['training_dir'])

        # Load the data
        train_dataloader, val_dataloader = data_pipeline(
            data_root=data_config['data_root'],
            data_source=data_config['data_source'],
            target_vars=data_config['target_vars'],
            time_column=data_config['time_column'],
            min_encoder_length=training_config['min_encoder_length'],
            max_encoder_length=training_config['max_encoder_length'],
            max_prediction_length=training_config['max_prediction_length'],
            min_prediction_length=training_config['min_prediction_length'],
            batch_size=training_config['batch_size'],
            num_workers=training_config['num_workers'],
            save_dir=data_config['save_dir']
        )

        # Train the model
        train_pipeline(train_dataloader, val_dataloader, config)

    elif args.mode == 'eval':
        evaluation_dir = create_evaluation_directory(base_dir=checkpoints_config['base_dir'], evaluation_subdir=checkpoints_config['evaluation_dir'])

        # Load the evaluation data
        eval_dataloader = data_pipeline(
            data_root=evaluation_config['eval_data_root'],
            data_source=data_config['data_source'],
            target_vars=data_config['target_vars'],
            time_column=data_config['time_column'],
            min_encoder_length=training_config['min_encoder_length'],
            max_encoder_length=training_config['max_encoder_length'],
            max_prediction_length=training_config['max_prediction_length'],
            min_prediction_length=training_config['min_prediction_length'],
            batch_size=training_config['batch_size'],
            num_workers=training_config['num_workers'],
            save_dir=evaluation_config['eval_save_dir']
        )[0]  # Only need the DataLoader for evaluation

        # Load the trained model
        best_model_path = os.path.join(training_dir, checkpoints_config['best_model_filename'])
        model = torch.load(best_model_path)

        # Evaluate the model
        evaluate_model(model, eval_dataloader, evaluation_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weather Forecasting with Temporal Fusion Transformer')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, help='Mode to run: train or eval')
    parser.add_argument('--config', type=str, default='configs/cds.yaml', help='Path to configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)