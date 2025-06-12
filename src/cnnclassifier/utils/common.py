import os,sys,shutil
from src.cnnclassifier import logger
import requests
from pathlib import Path
from box import Box
from src.cnnclassifier import CustomException
import yaml
from ensure import ensure_annotations
from typing import Any

import os
import sys
import json
import requests
from pathlib import Path
from ensure import ensure_annotations 


@ensure_annotations
def download_kaggle_dataset(api_url: str, local_filename: str = "data.zip", kaggle_json_path: str = "config/kaggle.json") -> str:
    try:
        # Load Kaggle API credentials
        with open(kaggle_json_path, "r") as f:
            kaggle_token = json.load(f)
        username = kaggle_token["username"]
        key = kaggle_token["key"]

        # Start authenticated session
        session = requests.Session()
        session.auth = (username, key)

        # Make initial request to get redirected URL
        response = session.get(api_url, allow_redirects=True, stream=True)
        response.raise_for_status()

        # Get file size for progress logging
        size = int(response.headers.get("Content-Length", 0))

        if os.path.exists(local_filename) and os.path.getsize(local_filename) == size:
            logger.info("Dataset already exists.")
            return local_filename

        # Download in chunks
        with open(local_filename, "wb") as f:
            tmp = 0
            last_log = -1
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    tmp += len(chunk)
                    total_percent = int((tmp / size) * 100) if size else 0
                    if total_percent % 10 == 0 and total_percent != last_log:
                        logger.info(f"Downloaded {total_percent}%")
                        last_log = total_percent

        logger.info(f"Download complete: {local_filename}")
        return local_filename

    except Exception as e:
        raise CustomException(e, sys)

    
    
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
    
    