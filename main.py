import argparse
from tools.data_process import data_pipeline
from datasets.cds.data_handling import create_cds_time_series_datasets
from tools.train import train_pipeline
from tools.eval import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Train and Evaluate Temporal Fusion Transformer Model')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, help='Mode of operation: `train` or `eval`.')
    parser.add_argument('--data_source', type=str, default='cds', help='The source of the data. Default is `cds`.')
    parser.add_argument('--data_root', type=str, default='data/samples/*.nc', help='The directory pattern to search for files.')
    parser.add_argument('--target_vars', type=str, nargs='+', default=["tcc", "hcc", "mcc", "lcc", "tciw", "tclw"], help='The list of target variables to include in the DataFrame.')
    parser.add_argument('--time_column', type=str, default='time', help='The name of the time column.')
    parser.add_argument('--max_encoder_length', type=int, default=365, help='The maximum length of the encoder.')
    parser.add_argument('--max_prediction_length', type=int, default=365, help='The maximum length of the prediction.')
    parser.add_argument('--min_prediction_length', type=int, default=1, help='The minimum length of the prediction.')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size for DataLoader.')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of workers for DataLoader.')
    parser.add_argument('--save_dir', type=str, default='', help='The directory to save the preprocessed data as `.csv`.')
    parser.add_argument('--model_path', type=str, default='model/checkpoints/best_model.ckpt', help='The path to the trained model checkpoint.')
    parser.add_argument('--new_data_root', type=str, default='', help='The directory pattern to search for new data files for prediction.')
    args = parser.parse_args()

    if args.mode == 'train':
        # Train the model
        train_dataloader, val_dataloader = data_pipeline(
            data_root=args.data_root,
            data_source=args.data_source,
            target_vars=args.target_vars,
            time_column=args.time_column,
            max_encoder_length=args.max_encoder_length,
            max_prediction_length=args.max_prediction_length,
            min_prediction_length=args.min_prediction_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_dir=args.save_dir
        )
        train_pipeline(train_dataloader, val_dataloader, param_tuning_trial_count=100)

    elif args.mode == 'eval':
        # Evaluate the model
        evaluate_model(
            model_path=args.model_path,
            data_root=args.data_root,
            target_vars=args.target_vars,
            time_column=args.time_column,
            max_encoder_length=args.max_encoder_length,
            max_prediction_length=args.max_prediction_length,
            min_prediction_length=args.min_prediction_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            new_data_root=args.new_data_root
        )

if __name__ == "__main__":
    main()
