from src.cnnclassifier.constants import *
from src.cnnclassifier.utils.common import read_yaml,create_directories
from src.cnnclassifier.entities.config_entity import *


class ConfigurationManager:
    def __init__(
        self,
        confif_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH) -> None:
        
        self.config = read_yaml(confif_filepath)
        self.params= read_yaml(params_filepath)
        print(type([self.config.artifacts_root]))
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        
        return data_ingestion_config