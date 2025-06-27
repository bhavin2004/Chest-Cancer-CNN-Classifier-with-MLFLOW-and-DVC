import sys
from src.cnnclassifier import logger,CustomException
from src.cnnclassifier.components.model_evaluation import Evaluation
from src.cnnclassifier.config.configuration import ConfigurationManager

STAGE_NAME = 'Evaluation State'

class EvaluationPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            eval_config = config.get_evaluation_config()
            evaluation = Evaluation(config=eval_config)
            evaluation.evaluaion()
            evaluation.log_into_mlflow()
            pass
        except Exception as e:
            CustomException(e,sys)
        
        
if __name__=="__main__":
    try:
        logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} started <<<<<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f"\n>>>>>>>>>> stage:{STAGE_NAME} completed <<<<<<<<<<\n\n{'='*50}")

    except Exception as e:
        raise CustomException(e,sys)
