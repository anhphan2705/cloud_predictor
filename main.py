from tools.data_preprocessing import data_pipeline

if __name__ == "__main__":
    data_root = 'data/samples/*.nc'
    data_pipeline(data_root)