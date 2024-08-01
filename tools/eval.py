import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from pytorch_forecasting import TemporalFusionTransformer, Baseline
from utils.file_utils import load_model
from utils.data_visualization import plot_predictions, interpret_model_predictions

def evaluate_loss(val_dataloader: DataLoader, model: TemporalFusionTransformer = Baseline, model_name: str = "Baseline") -> dict:
    """
    Evaluate model for loss comparison.

    Parameters:
    val_dataloader (DataLoader): DataLoader for the validation data.
    model (TemporalFusionTransformer): The trained Temporal Fusion Transformer model. Default is Baseline.
    model_name (str): The name of the model. Default is "Baseline".

    Returns:
    dict: The validation results for the model.
    """
    model = model.from_dataset(val_dataloader.dataset)
    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
    )
    val_result = trainer.validate(model, val_dataloader, verbose=False)
    print(f"[INFO] {model_name} model validation results: {val_result}")

    return val_result

# def evaluate_loss(val_dataloader: DataLoader, model = Baseline()) -> None:
#     """
#     Evaluate a baseline model for comparison.

#     Parameters:
#     val_dataloader (DataLoader): DataLoader for the validation data.
#     model (TemporalFusionTransformer): The trained Temporal Fusion Transformer model. Default is Baseline.

#     Returns:
#     Baseline: The baseline model.
#     """
#     predictions = model.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
#     print(f"[INFO] Baseline model validation results: {MAE()(predictions.output, predictions.y)}")

def perform_inference(model: TemporalFusionTransformer, dataloader: DataLoader, mode: str = 'raw', return_index: bool = True, return_x: bool = True, output_dir: str = None) -> dict:
    """
    Perform inference using the trained model.

    Parameters:
    model (TemporalFusionTransformer): The trained Temporal Fusion Transformer model.
    dataloader (DataLoader): DataLoader for the inference data.
    mode (str): Mode for prediction, can be 'prediction', 'raw', or 'prediction_interval'.
                - 'prediction': returns only the predictions.
                - 'raw': returns predictions along with additional information such as input features and indices.
                - 'quantiles': returns predictions for different quantiles, which is useful for uncertainty estimation.
    return_index (bool): Whether to return the prediction index in the same order as the output. Default is True.
    return_x (bool): Whether to return network inputs in the same order as the prediction output. Default is True.
    output_dir (str, optional): Directory to save the predictions. If None, predictions are not saved to a directory.

    Returns:
    dict: A dictionary containing the model predictions, with additional information such as prediction index and inputs, depending on the mode.
    """
    return model.predict(
        dataloader, 
        mode=mode, 
        return_index=return_index,  # return the prediction index in the same order as the output
        return_x=return_x,          # return network inputs in the same order as prediction output
        output_dir=output_dir,
        trainer_kwargs=dict(accelerator="gpu" if torch.cuda.is_available() else "cpu")
    )

def evaluate_pipeline(
    model_path: str,
    eval_dataloader: DataLoader,
    inference_dir: str,
    config: dict,
    show_future_observed: bool = False,
    add_loss_to_title: bool = False,
    show: bool = False
) -> None:
    """
    Evaluate the model by performing inference on the evaluation data and plot the predictions.

    Parameters:
    model_path (str): Path to the trained model checkpoint.
    eval_dataloader (DataLoader): DataLoader for the evaluation data.
    inference_dir (str): Directory to save the inference plots.
    config (dict): Dictionary containing configuration parameters.
    show_future_observed (bool, optional): If True, shows future observed values in the plots. Default is False.
    add_loss_to_title (bool, optional): If True, adds the loss to the plot titles. Default is False.
    show (bool, optional): If True, displays the plots. Default is False.
    """
    # Load the trained model
    model = load_model(model_path)
    print("[INFO] Model loaded successfully.")

    # Evaluate the baseline model compared to the trained model
    evaluate_loss(val_dataloader=eval_dataloader, model=Baseline, model_name="Baseline")
    evaluate_loss(val_dataloader=eval_dataloader, model=model, model_name="TFT")

    # Perform inference
    predictions = perform_inference(model, eval_dataloader, mode='raw', return_index=True, return_x=True)
    
    # Plot predictions
    plot_predictions(predictions, model=model, save_dir=inference_dir, show_future_observed=show_future_observed, add_loss_to_title=add_loss_to_title, show=show)
    print("[INFO] Model predictions plotted successfully")
    interpret_model_predictions(model, prediction=predictions, save_dir=inference_dir, model_name="tft", lags=config['time_series']['lags'], show=show)
    print("[INFO] Model predictions interpreted successfully")