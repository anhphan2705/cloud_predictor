import optuna
import torch
import pickle
from optuna.trial import Trial
from torch.utils.data import DataLoader
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

def objective(trial: Trial, train_dataloader: DataLoader, val_dataloader: DataLoader) -> float:
    """
    Objective function for Optuna hyperparameter tuning.

    Parameters:
    trial (optuna.trial.Trial): A trial object from Optuna for hyperparameter optimization.
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.

    Returns:
    float: Validation loss for the trial's set of hyperparameters.

    Example Usage:
    study.optimize(lambda trial: objective(trial, train_dataloader, val_dataloader), n_trials=100)
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 0.001, 0.1)
    hidden_size = trial.suggest_int("hidden_size", 8, 128)
    attention_head_size = trial.suggest_int("attention_head_size", 1, 4)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.3)
    hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 8, 128)

    # Create Temporal Fusion Transformer model
    tft = TemporalFusionTransformer.from_dataset(
        train_dataloader.dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=7,  # number of quantiles
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Define callbacks and logger
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    # Create PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        limit_train_batches=30,
        log_every_n_steps=10,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    # Train the model
    trainer.fit(tft, train_dataloader, val_dataloader)

    # Return the validation loss
    return trainer.callback_metrics["val_loss"].item()

def tune_hyperparameters(train_dataloader: DataLoader, val_dataloader: DataLoader, n_trials: int = 100) -> dict:
    """
    Tune hyperparameters using Optuna.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    n_trials (int): The number of trials for the hyperparameter tuning. Default is 100.

    Returns:
    dict: The best set of hyperparameters found by Optuna.

    Example Usage:
    best_params = tune_hyperparameters(train_dataloader, val_dataloader, n_trials=100)
    """
    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    
    # Optimize the study
    study.optimize(lambda trial: objective(trial, train_dataloader, val_dataloader), n_trials=n_trials)

    # Save the study results to a pickle file
    with open("study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # Get the best hyperparameters
    best_params = study.best_trial.params
    print(f"Best parameters: {best_params}")

    return best_params
