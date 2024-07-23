# Weather Forecasting using Temporal Fusion Transformer

This project aims to forecast weather conditions using the Temporal Fusion Transformer (TFT) model. The implementation leverages PyTorch Lightning for efficient model training, hyperparameter tuning, and logging. Additionally, the project includes baseline comparisons and visualization of predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project uses a Temporal Fusion Transformer (TFT) to perform time series forecasting for weather data. The TFT model is known for handling multiple time series with temporal relationships and incorporates various data sources efficiently.

## Installation
To install the necessary dependencies, run:
```bash
git clone https://github.com/anhphan2705/cloud_predictor.git
cd cloud_predictor
python -m venv tft_env
source tft_env/bin/activate  # On Windows, use `tft_env\Scripts\activate`
pip install -r requirements.txt
```

## Usage

### Datasets
You can set up your dataset 

### Configuration
Edit the `config.yaml` file to set the desired parameters for training, hyperparameter tuning, and logging. This file includes paths for data, model checkpoints, and logging directories, as well as hyperparameters for the TFT model.

### Training
To train the model, run:
```bash
python main.py --config config.yaml
```
This will start the training pipeline, which includes data loading, model initialization, hyperparameter tuning, and final training.

### Evaluation
The training pipeline evaluates the model against a baseline model. Validation metrics (loss, RMSE, MAE) are logged during training and evaluation.

### Visualization
After training, the script generates plots comparing the trained model's predictions with actual data and the baseline model's predictions. These plots are saved in the specified logs directory.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.