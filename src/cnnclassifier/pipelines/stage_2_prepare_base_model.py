import sys
from src.cnnclassifier import logger,CustomException
from src.cnnclassifier.components.prepare_base_model import PrepareBaseModel
# from src.cnnclassifier.config.configuration import ConfigurationManager

STAGE_NAME = 'Prepare Base Model State'

class PrepareBaseModelPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        # config = ConfigurationManager()
        # data_ingestion_config = config.get_data_ingestion_config()
        prepare_base_model = PrepareBaseModel()
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
        
        
if __name__=="__main__":
    try:
        logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} started <<<<<<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} completed <<<<<<<<<<\n\n{'='*50}")

    except Exception as e:
        raise CustomException(e,sys)
