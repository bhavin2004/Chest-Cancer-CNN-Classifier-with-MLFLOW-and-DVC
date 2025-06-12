import sys
from src.cnnclassifier import logger,CustomException
from src.cnnclassifier.components.data_ingestion import DataIngestion
# from src.cnnclassifier.config.configuration import ConfigurationManager

STAGE_NAME = 'Data Ingestion State'

class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        # config = ConfigurationManager()
        # data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion_obj = DataIngestion()
        data_ingestion_obj.download_file()
        data_ingestion_obj.unzip_dataset()
        
if __name__=="__main__":
    try:
        logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} started <<<<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} completed <<<<<<<<<<\n\n{'='*50}")

    except Exception as e:
            raise CustomException(e,sys)
