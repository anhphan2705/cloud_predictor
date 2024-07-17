import os
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from tools.hyperparam_tuning import tune_hyperparameters
from utils.file_utils import create_training_directory

def create_trainer(config, logger, checkpoint_callback, early_stop_callback, lr_logger, progress_bar):
    """
    Create a PyTorch Lightning trainer with specified configuration and callbacks.

    Parameters:
    config (dict): Dictionary containing training configuration parameters.
    logger (TensorBoardLogger): Logger for logging training process.
    checkpoint_callback (ModelCheckpoint): Callback for saving model checkpoints.
    early_stop_callback (EarlyStopping): Callback for early stopping.
    lr_logger (LearningRateMonitor): Callback for monitoring learning rate.
    progress_bar (TQDMProgressBar): Callback for progress bar.

    Returns:
    Trainer: The PyTorch Lightning trainer.
    """
    training_device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training device: {training_device}") 

    train_config = config['training']

    callbacks = [lr_logger, checkpoint_callback, early_stop_callback, progress_bar]
    for callback in callbacks:
        if callback is None:
            callbacks.remove(callback)

    return pl.Trainer(
        max_epochs=train_config['max_epochs'],
        accelerator=training_device,
        devices=1,
        gradient_clip_val=train_config['gradient_clip_val'],
        limit_train_batches=train_config['limit_train_batches'],
        log_every_n_steps=train_config['log_every_n_steps'],
        callbacks=callbacks,
        logger=logger,
    )

def initialize_model(train_dataloader, params, train_config, target_count):
    """
    Initialize the Temporal Fusion Transformer model from dataset and parameters.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    params (dict): Dictionary containing model parameters.
    config (dict): Dictionary containing training configuration parameters.

    Returns:
    TemporalFusionTransformer: Initialized Temporal Fusion Transformer model.
    """
    return TemporalFusionTransformer.from_dataset(
        train_dataloader.dataset,
        learning_rate=params.get("learning_rate", train_config['learning_rate']),
        hidden_size=params.get("hidden_size", train_config['hidden_size']),
        attention_head_size=params.get("attention_head_size", train_config['attention_head_size']),
        dropout=params.get("dropout", train_config['dropout']),
        hidden_continuous_size=params.get("hidden_continuous_size", train_config['hidden_continuous_size']),
        output_size= [7] * target_count,
        loss=QuantileLoss(),
        log_interval=train_config['log_every_n_steps'],
        reduce_on_plateau_patience=train_config['reduce_on_plateau_patience'],
    )

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
    training_dir, checkpoints_dir, logs_dir = create_training_directory(config['checkpoints']['base_dir'], config['checkpoints']['training_dir'])

    # Create model
    tft = initialize_model(train_dataloader, best_params, config['training'], len(config["data"]["target_vars"]))

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

    # Create trainer
    trainer = create_trainer(config, logger, checkpoint_callback, early_stop_callback, lr_logger, progress_bar)

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
        best_params = tune_hyperparameters(train_dataloader, val_dataloader, config, trainer_func=create_trainer, model_func=initialize_model)
    else:
        best_params = {}

    # Perform final training with the best hyperparameters
    trainer = training(train_dataloader, val_dataloader, best_params, config)

    # Load the best model w.r.t. the validation loss
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    return best_tft