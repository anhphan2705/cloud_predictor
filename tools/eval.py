import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_forecasting import TemporalFusionTransformer
from utils.file_utils import load_model, load_config
from utils.data_visualization import plot_predictions
from tools.data_process import data_pipeline

def perform_inference(model: TemporalFusionTransformer, dataloader: DataLoader) -> dict:
    """
    Perform inference using the trained model.

    Parameters:
    model (TemporalFusionTransformer): The trained Temporal Fusion Transformer model.
    dataloader (DataLoader): DataLoader for the inference data.

    Returns:
    dict: A dictionary containing the actual data and model predictions.
    """
    model_predictions = model.predict(
        dataloader, mode='raw', 
        return_index=True,  # return the prediction index in the same order as the output
        return_x=True,      # return network inputs in the same order as prediction output
        output_dir=None
    )
    
    return model_predictions

def evaluate_pipeline(model_path, eval_dataloader, inference_dir):

    # Load the trained model
    model = load_model(model_path)
    print("[INFO] Model loaded successfully.")

    # Perform inference
    predictions = perform_inference(model, eval_dataloader)
    print("[INFO] Inference completed")

    # Plot predictions
    print("[INFO] Plotting predictions...")
    plot_predictions(predictions, save_dir=inference_dir, show=True)