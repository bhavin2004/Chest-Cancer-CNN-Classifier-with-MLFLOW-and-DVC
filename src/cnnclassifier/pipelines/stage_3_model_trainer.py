import sys
from src.cnnclassifier import logger,CustomException
from src.cnnclassifier.components.model_trainer import Training

STAGE_NAME = 'Training State'

class TrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        # config = ConfigurationManager()
        # data_ingestion_config = config.get_data_ingestion_config()
        try:
            training = Training()
            training.train()
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__=="__main__":
    try:
        logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} started <<<<<<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} completed <<<<<<<<<<\n\n{'='*50}")

    except Exception as e:
        raise CustomException(e,sys)
