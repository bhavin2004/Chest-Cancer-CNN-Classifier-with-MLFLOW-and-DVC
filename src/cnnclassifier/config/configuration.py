from src.cnnclassifier.constants import *
from src.cnnclassifier.utils.common import read_yaml,create_directories
from src.cnnclassifier.entities.config_entity import *
import os



class ConfigurationManager:
    def __init__(
        self,
        confif_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH) -> None:
        
        self.config = read_yaml(confif_filepath)
        self.params= read_yaml(params_filepath)
        # print(type([self.config.artifacts_root]))
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
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params
        create_directories([config.root_dir])
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=params.IAMGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_classes=params.CLASSES,
            params_include_top=params.INCLUDE_TOP,
            params_weight=params.WEIGHTS,
            )
        
        return prepare_base_model_config
    
    def get_training_config(self) -> ModelTrainerConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params= self.params
        
        training_data = os.path.join(self.config.data_ingestion.unzip_dir,'Data')
        
        create_directories([Path(training.root_dir)])
        
        training_config = ModelTrainerConfig(
            root_dir=Path(training.root_dir),
            training_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_image_size=params.IAMGE_SIZE,
            params_is_augmentation=params.AUGMENTATION
            
        )
        
        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_to_model=self.config.training.trained_model_path,
            training_data=Path(os.path.join(self.config.data_ingestion.unzip_dir,'Data')),
            all_params=self.params,
            params_image_size=self.params.IAMGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )
        
        return eval_config