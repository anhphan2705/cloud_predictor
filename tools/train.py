from data_process import data_pipeline, dataloader
from datasets.cds.data_handling import create_time_series_datasets

targets=['tcc', 'hcc', 'mcc', 'lcc', 'tciw', 'tclw']

# Load the data
ts_dataset = data_pipeline(data_root='data/samples/*.nc', target_vars=['tcc', 'hcc', 'mcc', 'lcc', 'tciw', 'tclw'], min_prediction_length=24)
train_datasst, val_dataset = create_time_series_datasets(ts_dataset, max_encoder_length=365, max_prediction_length=365, targets=targets)
train_dataloader = dataloader(ts_dataset, train=True, batch_size=16, num_workers=4)
val_dataloader = dataloader(ts_dataset, train=False, batch_size=16, num_workers=4)