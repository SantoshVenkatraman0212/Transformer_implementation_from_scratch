'''
This file downloads the WMT 2014 English-German dataset as .zip using Kaggle API,
then extracts the train, test, and val sets inside /data/raw dir
'''
# Importing necessary libraries
import subprocess
from zipfile import ZipFile
from pathlib import Path
from config.paths import RAW_DATA_DIR, DATA_DIR


def download_and_extract() -> None:
    '''
    This file downloads the WMT-2014 dataset from kaggle as .zip, then extracts, and saves the dataset files
    '''
    # WMT 2014 Kaggle repo path
    dataset_repo_path = 'mohamedlotfy50/wmt-2014-english-german'
    # Creating the data dir if it doens't exist
    DATA_DIR.mkdir(parents = True, exist_ok = True)
    # Executing kaggle terminal download cmd as a separate subprocess
    # Check = True ensures that if the download fails it's raised as an error rather than failing silently
    subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_repo_path, '-p', DATA_DIR], check = True)
    zip_file_path = Path(f'{DATA_DIR / dataset_repo_path.rsplit('/', maxsplit = 1)[-1]}.zip')
    # Check if the zipfile was downloaded
    # If not raise File not found error
    if not zip_file_path.exists():
        raise(FileNotFoundError(f'The downloaded dataset zip file does not exist in {zip_file_path} path'))
    # Else read the file, and extract the contents to /data/raw path
    with ZipFile(zip_file_path, 'r') as zip_file:
        zip_file.extractall(RAW_DATA_DIR)
    # Unlinking from zip file path (Deleting the zip file)
    zip_file_path.unlink()


def main() -> None:
    '''
    Main function for running the download, and extraction script
    '''
    download_and_extract()

if __name__ == '__main__':
    main()

    
