import shutil
from dataclasses import dataclass
import sys
from pathlib import Path
from src.cnnclassifier import logger,CustomException
from src.cnnclassifier.utils.common import download_dataset
from src.cnnclassifier.config.configuration import ConfigurationManager



class DataIngestion:

    def __init__(self) -> None:
        self.config = ConfigurationManager().get_data_ingestion_config()
        
    def download_file(self):
        '''
        Fetch Data From Url
        '''
        
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            
            logger.info(f"Downloading dataset form {dataset_url} into file {zip_download_dir}")
            
            download_dataset(dataset_url,zip_download_dir)
            
            logger.info(f"Dataset Downloaded Successfully at {zip_download_dir}")
        
        except Exception as e:
            raise(CustomException(e,sys))
            
    def unzip_dataset(self):
        """it extracts the zip file
        """
        try:
            zip_file_path = self.config.local_data_file
            unzip_dir= self.config.unzip_dir
            logger.info(f"Starting to unzip the file {zip_file_path} at {unzip_dir}")
            shutil.unpack_archive(zip_file_path,unzip_dir)
            logger.info(f"Unzip Successfully at {unzip_dir}")
        except Exception as e:
            raise(CustomException(e,sys))
        