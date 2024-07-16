import os
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from tools.hyperparam_tuning import tune_hyperparameters
from utils.file_utils import create_training_directory

def training(train_dataloader: DataLoader, val_dataloader: DataLoader, best_params: dict, config: dict) -> pl.Trainer:
    """
    Perform the final training of the Temporal Fusion Transformer using the best hyperparameters.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    best_params (dict): Dictionary containing the best hyperparameters.
    config (dict): Dictionary containing training configuration parameters.

    Returns:
    Trainer: The trained PyTorch Lightning trainer.

    Example Usage:
    trainer = final_training(train_dataloader, val_dataloader, best_params, config)
    """
    # Create directories for checkpoints and logs
    training_dir, checkpoints_dir, logs_dir = create_training_directory()

    # Create Temporal Fusion Transformer model
    tft = TemporalFusionTransformer.from_dataset(
        train_dataloader.dataset,
        learning_rate=best_params.get("learning_rate", config['training']['learning_rate']),
        hidden_size=best_params.get("hidden_size", config['training']['hidden_size']),
        attention_head_size=best_params.get("attention_head_size", config['training']['attention_head_size']),
        dropout=best_params.get("dropout", config['training']['dropout']),
        hidden_continuous_size=best_params.get("hidden_continuous_size", config['training']['hidden_continuous_size']),
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=config['training']['reduce_on_plateau_patience'],
    )

    # Define callbacks and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=config['checkpoints']['checkpoint_filename'],
        save_top_k=-1,  # Save all checkpoints
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=config['training']['early_stop_min_delta'], patience=config['training']['early_stop_patience'], verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    progress_bar = TQDMProgressBar(refresh_rate=1)
    logger = TensorBoardLogger(save_dir=logs_dir, name=config['logging']['log_name'])

    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=config['training']['gradient_clip_val'],
        limit_train_batches=config['training']['limit_train_batches'],
        log_every_n_steps=config['training']['log_every_n_steps'],
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback, progress_bar],
        logger=logger,
    )

    print(f"[INFO] Loaded model with {tft.size()} parameters.\n{tft}")
    print(f"[INFO] Starting training...")

    # Train the model
    trainer.fit(tft, train_dataloader, val_dataloader)

    # Save the best model
    best_model_path = checkpoint_callback.best_model_path
    final_best_model_path = os.path.join(training_dir, config['checkpoints']['best_model_filename'])
    if best_model_path:
        torch.save(torch.load(best_model_path), final_best_model_path)

    return trainer

def train_pipeline(train_dataloader: DataLoader, val_dataloader: DataLoader, config: dict) -> TemporalFusionTransformer:
    """
    Execute the training pipeline, including hyperparameter tuning and final training.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    config (dict): Dictionary containing training and hyperparameter tuning configuration parameters.

    Returns:
    TemporalFusionTransformer: The trained Temporal Fusion Transformer model.

    Example Usage:
    train_pipeline(train_dataloader, val_dataloader, config)
    """
    if config['hyperparameter_tuning']['enable']:
        # Tune hyperparameters
        best_params = tune_hyperparameters(train_dataloader, val_dataloader, n_trials=config['hyperparameter_tuning']['n_trials'])
    else:
        best_params = {}

    # Perform final training with the best hyperparameters
    trainer = training(train_dataloader, val_dataloader, best_params, config)

    # Load the best model w.r.t. the validation loss
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    return best_tft