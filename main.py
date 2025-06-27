from src.cnnclassifier import logger,CustomException
import sys
from src.cnnclassifier.pipelines.stage_1_data_ingestion import DataIngestionTrainingPipeline
from src.cnnclassifier.pipelines.stage_2_prepare_base_model import PrepareBaseModelPipeline
from src.cnnclassifier.pipelines.stage_3_model_trainer import TrainingPipeline
from src.cnnclassifier.pipelines.stage_4_model_evaluation import EvaluationPipeline
from src.cnnclassifier.config.configuration import ConfigurationManager
import tensorflow as tf 
import warnings

warnings.filterwarnings("ignore")


STAGE_NAME = 'Data Ingestion State'

try:
    configure_obj= ConfigurationManager
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




STAGE_NAME = 'Training State'
try:
    logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} started <<<<<<<<<<")
    print("Using GPU:", tf.config.list_logical_devices('GPU'))
    obj = TrainingPipeline()
    obj.main()
    logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} completed <<<<<<<<<<\n\n{'='*50}")

except Exception as e:
    raise CustomException(e,sys)

STAGE_NAME = 'Evaluation State'
try:
    logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} started <<<<<<<<<<")
    print("Using GPU:", tf.config.list_logical_devices('GPU'))
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} completed <<<<<<<<<<\n\n{'='*50}")

except Exception as e:
    raise CustomException(e,sys)