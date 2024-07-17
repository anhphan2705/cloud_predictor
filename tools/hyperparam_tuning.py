import optuna
import torch
import pickle
from optuna.trial import Trial
from torch.utils.data import DataLoader
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

def objective(trial: Trial, train_dataloader: DataLoader, val_dataloader: DataLoader, **hyperparameter_tuning_config) -> float:
    """
    Objective function for Optuna hyperparameter tuning.

    Parameters:
    trial (optuna.trial.Trial): A trial object from Optuna for hyperparameter optimization.
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    **hyperparameter_tuning_config: Additional training configurations.

    Returns:
    float: Validation loss for the trial's set of hyperparameters.

    Example Usage:
    study.optimize(lambda trial: objective(trial, train_dataloader, val_dataloader, **hyperparameter_tuning_config), n_trials=100)
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", *hyperparameter_tuning_config["learning_rate_range"])
    hidden_size = trial.suggest_int("hidden_size", *hyperparameter_tuning_config["hidden_size_range"])
    attention_head_size = trial.suggest_int("attention_head_size", *hyperparameter_tuning_config["attention_head_size_range"])
    dropout = trial.suggest_uniform("dropout", *hyperparameter_tuning_config["dropout_range"])
    hidden_continuous_size = trial.suggest_int("hidden_continuous_size", *hyperparameter_tuning_config["hidden_continuous_size_range"])

    # Create Temporal Fusion Transformer model
    tft = TemporalFusionTransformer.from_dataset(
        train_dataloader.dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=[7],  # number of quantiles
        loss=QuantileLoss(),
        log_interval=hyperparameter_tuning_config['log_every_n_steps'],
        reduce_on_plateau_patience=hyperparameter_tuning_config['reduce_on_plateau_patience'],
    )

    # Define callbacks and logger
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=hyperparameter_tuning_config['early_stop_min_delta'], patience=hyperparameter_tuning_config['early_stop_patience'], verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    training_device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training devide: {training_device}")

    # Create PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=hyperparameter_tuning_config['max_epochs'],
        accelerator=training_device,
        devices=1,
        gradient_clip_val=hyperparameter_tuning_config['gradient_clip_val'],
        limit_train_batches=hyperparameter_tuning_config['limit_train_batches'],
        log_every_n_steps=hyperparameter_tuning_config['log_every_n_steps'],
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    # Train the model
    trainer.fit(tft, train_dataloader, val_dataloader)

    # Return the validation loss
    return trainer.callback_metrics["val_loss"].item()

def tune_hyperparameters(train_dataloader: DataLoader, val_dataloader: DataLoader, **hyperparameter_tuning_config) -> dict:
    """
    Tune hyperparameters using Optuna.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    n_trials (int): The number of trials for the hyperparameter tuning. Default is 100.
    **hyperparameter_tuning_config: Additional training configurations.

    Returns:
    dict: The best set of hyperparameters found by Optuna.

    Example Usage:
    best_params = tune_hyperparameters(train_dataloader, val_dataloader, n_trials=100, **hyperparameter_tuning_config)
    """
    print("[INFO] Tuning hyperparameters using Optuna...")
    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    
    # Optimize the study
    study.optimize(lambda trial: objective(trial, train_dataloader, val_dataloader, **hyperparameter_tuning_config))

    # Save the study results to a pickle file
    with open("study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # Get the best hyperparameters
    best_params = study.best_trial.params
    print(f"[INFO] Best parameters: {best_params}")

    return best_params