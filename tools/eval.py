import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_forecasting import Baseline, TemporalFusionTransformer
from utils.file_utils import load_model
from utils.data_visualization import plot_predictions, interpret_model_predictions

def get_predictions(trained_model: TemporalFusionTransformer, baseline_model: Baseline, dataloader: DataLoader) -> dict:
    """
    Get predictions from the trained model and the baseline model.

    Parameters:
    trained_model (TemporalFusionTransformer): The trained Temporal Fusion Transformer model.
    baseline_model (Baseline): The baseline model.
    dataloader (DataLoader): DataLoader for the validation data.

    Returns:
    dict: A dictionary containing the actual data, trained model predictions, and baseline model predictions.
    """
    actuals = torch.cat([y[0] for x, y in iter(dataloader)]).cpu().numpy()
    trained_model_predictions = trained_model.predict(dataloader).cpu().numpy()
    baseline_model_predictions = baseline_model.predict(dataloader).cpu().numpy()
    
    return {
        "actuals": actuals,
        "trained_model_predictions": trained_model_predictions,
        "baseline_model_predictions": baseline_model_predictions
    }

def perform_inference(model: TemporalFusionTransformer, dataloader: DataLoader) -> dict:
    """
    Perform inference using the trained model.

    Parameters:
    model (TemporalFusionTransformer): The trained Temporal Fusion Transformer model.
    dataloader (DataLoader): DataLoader for the inference data.

    Returns:
    dict: A dictionary containing the model predictions, with additional information such as prediction index and inputs.
    """
    model_predictions = model.predict(
        dataloader, mode='raw', 
        return_index=True,  # return the prediction index in the same order as the output
        return_x=True,      # return network inputs in the same order as prediction output
        output_dir=None
    )
    
    return model_predictions

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

    # Perform inference
    predictions = perform_inference(model, eval_dataloader)
    print("[INFO] Inference completed")

    interpret_model_predictions(model, eval_dataloader, save_dir=inference_dir, model_name="tft", lags=config['time_series']['lags'], prediction=predictions, show=False)
    print("[INFO] Model predictions interpreted successfully")

    # Plot predictions
    print("[INFO] Plotting predictions...")
    plot_predictions(predictions, model=model, save_dir=inference_dir, show_future_observed=show_future_observed, add_loss_to_title=add_loss_to_title, show=show)