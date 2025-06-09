from src.cnnclassifier import logger,CustomException
import sys
from src.cnnclassifier.pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = 'Data Ingestion State'

try:
    logger.info(f">>>>>>>>>> stage:{STAGE_NAME} started <<<<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>>>> stage:{STAGE_NAME} completed <<<<<<<<<<\n\n{'='*50}")

except Exception as e:
        raise CustomException(e,sys)