import os
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import Baseline, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss
from tools.hyperparam_tuning import tune_hyperparameters
from utils.file_utils import create_training_directory
from utils.data_visualization import plot_predictions

def get_predictions(trained_model, baseline_model, val_dataloader):
    """
    Get predictions from the trained model and the baseline model.

    Parameters:
    trained_model (TemporalFusionTransformer): The trained Temporal Fusion Transformer model.
    baseline_model (Baseline): The baseline model.
    val_dataloader (DataLoader): DataLoader for the validation data.

    Returns:
    dict: A dictionary containing the actual data, trained model predictions, and baseline model predictions.
    """
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    trained_model_predictions = trained_model.predict(val_dataloader)
    baseline_model_predictions = baseline_model.predict(val_dataloader)
    
    return {
        "actuals": actuals,
        "trained_model_predictions": trained_model_predictions,
        "baseline_model_predictions": baseline_model_predictions
    }

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
    Trainer: The PyTorch Lightning trainer.
    """
    training_device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training device: {training_device}") 

    train_config = config['training']

    callbacks = [lr_logger, checkpoint_callback, early_stop_callback, progress_bar]
    callbacks = [callback for callback in callbacks if callback is not None]

    return pl.Trainer(
        max_epochs=train_config['max_epochs'],
        accelerator=training_device,
        strategy='ddp_spawn',  # For Windows users, comment out if Linux
        devices=1,
        gradient_clip_val=train_config['gradient_clip_val'],
        limit_train_batches=train_config['limit_train_batches'],
        log_every_n_steps=train_config['log_every_n_steps'],
        callbacks=callbacks,
        logger=logger,
    )

def initialize_model(train_dataloader: DataLoader, params: dict, train_config: dict, target_count: int):
    """
    Initialize the Temporal Fusion Transformer model from dataset and parameters.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    params (dict): Dictionary containing model parameters.
    train_config (dict): Dictionary containing training configuration parameters.
    target_count (int): The number of target variables.

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
        output_size=[7] * target_count,
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
    training_dir, checkpoints_dir, logs_dir = create_training_directory(config['checkpoints']['base_dir'], config['checkpoints']['training_dir'])
    tft = initialize_model(train_dataloader, best_params, config['training'], len(config["data"]["target_vars"]))

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=config['checkpoints']['checkpoint_filename'],
        save_top_k=-1,
        monitor='val_loss',
        mode='min'
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
    final_best_model_path = os.path.join(training_dir, config['checkpoints']['best_model_filename'])

    if best_model_path:
        torch.save(torch.load(best_model_path), final_best_model_path)

    return trainer

def evaluate_baseline(train_dataloader: DataLoader, val_dataloader: DataLoader, config: dict) -> Baseline:
    """
    Evaluate a baseline model for comparison.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    config (dict): Dictionary containing configuration parameters.

    Returns:
    Baseline: The baseline model.
    """
    baseline_model = Baseline.from_dataset(train_dataloader.dataset)
    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
    )
    val_result = trainer.validate(baseline_model, val_dataloader, verbose=False)
    print(f"[INFO] Baseline model validation results: {val_result}")

    return baseline_model

def train_pipeline(train_dataloader: DataLoader, val_dataloader: DataLoader, logs_dir: str, config: dict) -> TemporalFusionTransformer:
    """
    Execute the training pipeline, including hyperparameter tuning and final training.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    logs_dir (str): The directory to save logs.
    config (dict): Dictionary containing training and hyperparameter tuning configuration parameters.

    Returns:
    TemporalFusionTransformer: The trained Temporal Fusion Transformer model.

    Example Usage:
    train_pipeline(train_dataloader, val_dataloader, config)
    """
    if config['hyperparameter_tuning']['enable']:
        best_params = tune_hyperparameters(train_dataloader, val_dataloader, logs_dir, config, trainer_func=create_trainer, model_func=initialize_model)
    else:
        best_params = {}

    trainer = training(train_dataloader, val_dataloader, best_params, config)

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    baseline_model = evaluate_baseline(train_dataloader, val_dataloader, config)

    predictions = get_predictions(best_tft, baseline_model, val_dataloader)

    plot_predictions(predictions, save_dir=logs_dir)

    return best_tft