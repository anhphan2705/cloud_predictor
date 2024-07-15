from tools.data_process import data_pipeline

if __name__ == "__main__":
    data_root = 'data/samples/*.nc'
    data_pipeline(data_root)