import optuna
import pickle
from optuna.trial import Trial
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

def objective(trial: Trial, train_dataloader: DataLoader, val_dataloader: DataLoader, config: dict, trainer_func: callable, model_func: callable) -> float:
    """
    Objective function for Optuna hyperparameter tuning.

    Parameters:
    trial (optuna.trial.Trial): A trial object from Optuna for hyperparameter optimization.
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    config (dict): Dictionary containing configuration parameters.
    trainer_func (callable): Function to create a PyTorch Lightning trainer.
    model_func (callable): Function to initialize the Temporal Fusion Transformer model.

    Returns:
    float: Validation loss for the trial's set of hyperparameters.
    """
    training_config = config["training"]
    hyperparameter_tuning_config = config["hyperparameter_tuning"]

    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", *hyperparameter_tuning_config["learning_rate_range"], log=True)
    hidden_size = trial.suggest_int("hidden_size", *hyperparameter_tuning_config["hidden_size_range"])
    attention_head_size = trial.suggest_int("attention_head_size", *hyperparameter_tuning_config["attention_head_size_range"])
    dropout = trial.suggest_float("dropout", *hyperparameter_tuning_config["dropout_range"])
    hidden_continuous_size = trial.suggest_int("hidden_continuous_size", *hyperparameter_tuning_config["hidden_continuous_size_range"])

    params = {
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "attention_head_size": attention_head_size,
        "dropout": dropout,
        "hidden_continuous_size": hidden_continuous_size,
    }

    # Create model
    tft = model_func(train_dataloader, params, training_config)

    # Define callbacks and logger
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=hyperparameter_tuning_config['early_stop_min_delta'], patience=hyperparameter_tuning_config['early_stop_patience'], verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")
    progress_bar = TQDMProgressBar(refresh_rate=1)

    # Create trainer
    trainer = trainer_func(config, logger, None, early_stop_callback, lr_logger, progress_bar)

    # Train the model
    trainer.fit(tft, train_dataloader, val_dataloader)

    # Return the validation loss
    return trainer.callback_metrics["val_loss"].item()

def tune_hyperparameters(train_dataloader: DataLoader, val_dataloader: DataLoader, config: dict, trainer_func: callable, model_func: callable) -> dict:
    """
    Tune hyperparameters using Optuna.

    Parameters:
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    config (dict): Dictionary containing configuration parameters.
    trainer_func (callable): Function to create a PyTorch Lightning trainer.
    model_func (callable): Function to initialize the Temporal Fusion Transformer model.

    Returns:
    dict: The best set of hyperparameters found by Optuna.
    """
    print("[INFO] Tuning hyperparameters using Optuna...")
    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    
    # Optimize the study
    study.optimize(lambda trial: objective(trial, train_dataloader, val_dataloader, config, trainer_func, model_func))

    # Save the study results to a pickle file
    with open("study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # Get the best hyperparameters
    best_params = study.best_trial.params
    print(f"[INFO] Best parameters: {best_params}")

    return best_params