# weather_forecast

Combining Temporal Fusion Transformers (TFT) with Reinforcement Learning (RL) is an advanced approach that could potentially leverage the sequential pattern recognition of TFTs and the decision-making capabilities of RL.

### Conceptual Overview:

1. **TFT for Feature Extraction:**
   - Use TFT to process historical weather data and extract relevant temporal features and patterns.
   - The TFT model can act as a sophisticated feature extractor, capturing complex relationships in the data.

2. **RL for Decision Making:**
   - Use the extracted features from the TFT as input to an RL agent.
   - The RL agent can learn to make decisions or predictions based on these features, optimizing for specific objectives or rewards.

### Steps to Combine TFT with RL:

#### 1. Data Preparation:
- **Historical Weather Data:**
  - Gather and preprocess historical weather data.
  - Split the data into training, validation, and test sets.

#### 2. Train the TFT Model:
- **Architecture Setup:**
  - Define and configure the TFT model architecture.
  - Include relevant covariates (static and dynamic features) for the prediction task.
- **Training:**
  - Train the TFT model on the historical weather data to learn temporal patterns and feature importance.
- **Feature Extraction:**
  - Once trained, use the TFT model to extract features from the weather data. These features will serve as input for the RL agent.

#### 3. Integrate with Reinforcement Learning:
- **RL Environment:**
  - Define the RL environment. The state can include features extracted by the TFT, while the action space could involve making predictions or other relevant decisions.
- **Reward Function:**
  - Design a reward function that optimize prediction accuracy, minimizing prediction error.
- **RL Agent:**
  - Need experiment to decide which is the best RL agent (PPO, DeepQ, ...)
  - Train the RL agent using the features provided by the TFT model as inputs to learn the optimal policy.

#### 4. Training the Combined Model:
- **Sequential Training:**
  - First, train the TFT model independently on historical data.
  - Use the trained TFT model to extract features and then train the RL agent on these features.
- **End-to-End Training:**
  - For a sophisticated implementations, end-to-end training pipeline needed to be setup where the TFT and RL models are trained jointly. This requires careful management of the training loop and potentially custom loss functions.

#### 5. Evaluation and Tuning:
- **Validation:**
  - Evaluate the combined model on validation data to ensure it generalizes well.
- **Hyperparameter Tuning:**
  - Tune hyperparameters for both the TFT and RL components to optimize performance.

### Workflow:

1. **Data Collection and Preprocessing:**
   - Gather historical weather data and preprocess it for model training.
   
2. **TFT Training:**
   - Configure and train the TFT model to learn temporal patterns and extract features.

3. **Feature Extraction:**
   - Use the trained TFT model to generate features from historical data.

4. **RL Environment and Training:**
   - Define the RL environment and reward function.
   - Train the RL agent using the features extracted by the TFT model.

5. **Evaluation and Fine-Tuning:**
   - Evaluate the combined model on a test set.
   - Perform hyperparameter tuning and iterative improvements.

### Planning Folder Structure

```
weather_forecast/
│
├── data/
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data files ready for model input
│   ├── train/              # Training data
│   ├── val/                # Validation data
│   ├── test/               # Test data
│   └── external/           # External data sources from APIs
│
├── notebooks/
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   ├── preprocessing.ipynb # Data preprocessing and feature engineering
│   └── training.ipynb      # Model training and evaluation
│
├── tools/
│   ├── data_preprocessing.py # Data preprocessing scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── utils.py              # Utility functions
│
├── models/
│   ├── tft_model.py         # TFT model definition
│   ├── checkpoints/         # Saved model checkpoints
│   └── __init__.py          # Init file to make this directory a package
│
├── config/
│   ├── config.yaml          # Configuration files for hyperparameters, paths, etc.
│   └── __init__.py          # Init file to make this directory a package
│
├── logs/
│   ├── training_logs/       # Logs for training runs
│   └── evaluation_logs/     # Logs for evaluation runs
│
├── experiments/
│   └── ...                  # Experiments
│
├── venv/
│   └── ...
│
├── requirements.txt         # List of dependencies
├── README.md                # Project description and setup instructions
└── .gitignore               # Git ignore file
```