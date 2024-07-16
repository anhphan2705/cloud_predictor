import os
from datetime import datetime
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from hyperparam_tuning import tune_hyperparameters

def training(train_dataloader: DataLoader, val_dataloader: DataLoader, best_params: dict) -> pl.Trainer:
    """
    Perform the final training of the Temporal Fusion Transformer using the best hyperparameters.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    best_params (dict): Dictionary containing the best hyperparameters.

    Returns:
    Trainer: The trained PyTorch Lightning trainer.

    Example Usage:
    trainer = final_training(train_dataloader, val_dataloader, best_params)
    """
    # Create Temporal Fusion Transformer model
    tft = TemporalFusionTransformer.from_dataset(
        train_dataloader.dataset,
        learning_rate=best_params["learning_rate"],
        hidden_size=best_params["hidden_size"],
        attention_head_size=best_params["attention_head_size"],
        dropout=best_params["dropout"],
        hidden_continuous_size=best_params["hidden_continuous_size"],
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Define callbacks and logger
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    progress_bar = TQDMProgressBar(refresh_rate=10)
    logger = TensorBoardLogger("lightning_logs")

    # Define checkpoint callbacks
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./model/checkpoints/{current_time}",
        filename="epoch{epoch:02d}",
        save_top_k=-1,  # Save all checkpoints
        save_last=False,
        every_n_epochs=10,  # Save every 10 epochs
    )

    best_model_callback = ModelCheckpoint(
        dirpath=f"./model",
        filename=f"model_{current_time}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,  # Only save the best model
    )

    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        limit_train_batches=30,
        log_every_n_steps=10,
        callbacks=[lr_logger, early_stop_callback, progress_bar, checkpoint_callback, best_model_callback],
        logger=logger,
    )

    # Train the model
    trainer.fit(tft, train_dataloader, val_dataloader)

    return trainer

def train_pipeline(train_dataloader: DataLoader, val_dataloader: DataLoader, param_tuning_trial_count: int = 100) -> TemporalFusionTransformer:
    """
    Execute the training pipeline, including hyperparameter tuning and final training.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    param_tuning_trial_count (int): Number of trials for hyperparameter tuning. Default is 100.

    Returns:
    TemporalFusionTransformer: The trained Temporal Fusion Transformer model.

    Example Usage:
    train_pipeline(train_dataloader, val_dataloader, param_tuning_trial_count=100)
    """
    # Tune hyperparameters
    best_params = tune_hyperparameters(train_dataloader, val_dataloader, n_trials=param_tuning_trial_count)

    # Perform final training with the best hyperparameters
    trainer = training(train_dataloader, val_dataloader, best_params)

    # Load the best model w.r.t. the validation loss
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    return best_tft