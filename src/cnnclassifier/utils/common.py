import os,sys,shutil
from src.cnnclassifier import logger
import requests
from pathlib import Path
from box import Box
from src.cnnclassifier import CustomException
import yaml
from ensure import ensure_annotations
from typing import Any


@ensure_annotations
def download_dataset(url:str, local_filename:str='data.zip'):
    try:
        size = int(requests.head(url).headers['Content-Length'])
        if (os.path.exists(local_filename)) and (os.path.getsize(local_filename)==size):
            logger.info("Dataset Already Exists")

        else:    
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    tmp=0
                    last_log=-1
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
                        tmp+=len(chunk)
                        
                        total_download = int((tmp/size) * 100) if size else 0 
                        if total_download%10==0 and total_download!=last_log:
                            logger.info(f"Downloaded {total_download}% of Dataset ")
                            last_log = total_download
        return local_filename
    except Exception as e:
        raise(CustomException(e,sys))
    
    
@ensure_annotations
def read_yaml(yaml_filepath: Path) -> Box:
    """Used to read the yaml file and create Box object

    Args:
        yaml_filepath (Path): Path to the yaml file

    Returns:
        Box: Dictionary type object that contain yaml files data
    """
    try:
        with open(yaml_filepath) as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
            logger.info(f'yaml file:{yaml_filepath} loaded successfully')
            
            return Box(yaml_data)
    except Exception as e:
        raise(CustomException(e,sys))
    
@ensure_annotations
def create_directories(path_to_directories:list,verbose=True):
    """Used to create directories 

    Args:
        path_to_directories (list): list containd directories
        verbose (bool, optional): to log the creation of directories. Defaults to True.
    """
    try:
        for path in path_to_directories:
            os.makedirs(path,exist_ok=True)
            if verbose:
                logger.info(f"{path} directory is created")
    except Exception as e:
        raise(CustomException(e,sys))
    
    