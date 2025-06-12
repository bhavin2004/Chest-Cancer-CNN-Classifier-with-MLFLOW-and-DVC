from src.cnnclassifier import logger,CustomException
import sys
from src.cnnclassifier.pipelines.stage_1_data_ingestion import DataIngestionTrainingPipeline
from src.cnnclassifier.pipelines.stage_2_prepare_base_model import PrepareBaseModelPipeline

STAGE_NAME = 'Data Ingestion State'

try:
    logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} started <<<<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} completed <<<<<<<<<<\n\n{'='*50}")

except Exception as e:
        raise CustomException(e,sys)
    


STAGE_NAME = 'Prepare Base Model State'
try:
    logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} started <<<<<<<<<<")
    obj = PrepareBaseModelPipeline()
    obj.main()
    logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} completed <<<<<<<<<<\n\n{'='*50}")

except Exception as e:
    raise CustomException(e,sys)