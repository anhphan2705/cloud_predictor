import os
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from tools.hyperparam_tuning import tune_hyperparameters
from tools.eval import evaluate_pipeline

def create_trainer(config: dict, logger: TensorBoardLogger, checkpoint_callback: ModelCheckpoint, early_stop_callback: EarlyStopping, lr_logger: LearningRateMonitor, progress_bar: TQDMProgressBar) -> pl.Trainer:
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
    pl.Trainer: The PyTorch Lightning trainer.
    """
    training_device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training device: {training_device}") 

    train_config = config['training']

    callbacks = [lr_logger, checkpoint_callback, early_stop_callback, progress_bar]
    callbacks = [callback for callback in callbacks if callback is not None]

    return pl.Trainer(
        max_epochs=train_config['max_epochs'],
        accelerator=training_device,
        # strategy=DDPStrategy(process_group_backend="gloo"),  # For Windows users, comment out if Linux
        devices=1,
        gradient_clip_val=train_config['gradient_clip_val'],
        limit_train_batches=train_config['limit_train_batches'],
        log_every_n_steps=train_config['log_every_n_steps'],
        callbacks=callbacks,
        logger=logger,
    )

def initialize_model(train_dataloader: DataLoader, params: dict, train_config: dict, target_count: int = 1) -> TemporalFusionTransformer:
    """
    Initialize the Temporal Fusion Transformer model from dataset and parameters.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    params (dict): Dictionary containing model parameters.
    train_config (dict): Dictionary containing training configuration parameters.
    target_count (int): The number of target variables. Default is 1.

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
        output_size=train_config['output_size'] if target_count == 1 else [train_config['output_size']] * target_count,
        loss=QuantileLoss(),
        log_interval=train_config['log_every_n_steps'],
        reduce_on_plateau_patience=train_config['reduce_on_plateau_patience'],
    )

def training(train_dataloader: DataLoader, val_dataloader: DataLoader, best_params: dict, training_dir: str, checkpoint_dir: str, logs_dir: str, config: dict) -> pl.Trainer:
    """
    Perform the final training of the Temporal Fusion Transformer using the best hyperparameters.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    best_params (dict): Dictionary containing the best hyperparameters.
    training_dir (str): Directory to save the training results.
    checkpoint_dir (str): Directory to save the checkpoints.
    logs_dir (str): Directory to save the logs.
    config (dict): Dictionary containing training configuration parameters.

    Returns:
    pl.Trainer: The trained PyTorch Lightning trainer.
    """
    tft = initialize_model(train_dataloader, best_params, config['training'], target_count=len(config['time_series']['target_vars']))

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=config['checkpoint']['checkpoint_filename'],
        save_top_k=config['checkpoint']['save_top_k'],
        every_n_epochs=config['checkpoint']['every_n_epochs'],
        monitor=config['checkpoint']['monitor'],
        mode=config['checkpoint']['mode'],
        save_last=config['checkpoint']['save_last'],
        save_weights_only=config['checkpoint']['save_weights_only'],
        verbose=config['checkpoint']['verbose']
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=config['training']['early_stop_min_delta'], patience=config['training']['early_stop_patience'], verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    progress_bar = TQDMProgressBar(refresh_rate=1)
    logger = TensorBoardLogger(save_dir=logs_dir, name="training_logs")

    trainer = create_trainer(config, logger, checkpoint_callback, early_stop_callback, lr_logger, progress_bar)

    print(f"[INFO] Loaded model with {tft.size()} parameters.\n{tft}")
    print(f"[INFO] Starting training...")

    trainer.fit(tft, train_dataloader, val_dataloader)

    best_model_path = checkpoint_callback.best_model_path
    final_best_model_path = os.path.join(training_dir, config['checkpoint']['best_model_filename'])

    if best_model_path:
        torch.save(torch.load(best_model_path), final_best_model_path)

    return trainer

def train_pipeline(train_dataloader: DataLoader, val_dataloader: DataLoader, training_dir: str, checkpoint_dir: str, logs_dir: str, inference_dir: str, config: dict) -> TemporalFusionTransformer:
    """
    Execute the training pipeline, including hyperparameter tuning and final training.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    training_dir (str): Directory to save the training results.
    checkpoint_dir (str): Directory to save the checkpoints.
    logs_dir (str): Directory to save the logs.
    inference_dir (str): Directory to save the inference results.
    config (dict): Dictionary containing training and hyperparameter tuning configuration parameters.

    Returns:
    TemporalFusionTransformer: The trained Temporal Fusion Transformer model.
    """
    if config['hyperparameter_tuning']['enable']:
        best_params = tune_hyperparameters(train_dataloader, val_dataloader, logs_dir, config, trainer_func=create_trainer, model_func=initialize_model)
    else:
        best_params = {}

    trainer = training(train_dataloader, val_dataloader, best_params, training_dir, checkpoint_dir, logs_dir, config)

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    evaluate_pipeline(best_model_path, val_dataloader, inference_dir, config=config, show_future_observed=True, add_loss_to_title=True, show=False)

    return best_tft