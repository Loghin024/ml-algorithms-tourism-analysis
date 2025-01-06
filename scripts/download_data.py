from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate
api = KaggleApi()
api.authenticate()

dataset_name = "umeradnaan/tourism-dataset"
download_path = "../data"

try:
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print(f"Dataset successfully downloaded and extracted to: {download_path}")
except Exception as e:
    print(f"ERROR: {e}")
