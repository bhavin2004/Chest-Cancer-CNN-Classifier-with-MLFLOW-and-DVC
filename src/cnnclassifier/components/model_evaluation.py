import tensorflow as tf
import mlflow
import dagshub
import mlflow.keras
import dagshub
from urllib.parse import urlparse
from src.cnnclassifier.entities.config_entity import EvaluationConfig
import os
from pathlib import Path
from tensorflow.keras.applications.vgg16 import preprocess_input
from src.cnnclassifier.utils.common import save_json



class Evaluation:
    def __init__(self,config=EvaluationConfig) :
        self.config=config
    
    def _valid_generator(self):
        
        datagenerator_kwargs = dict(preprocessing_function=preprocess_input)
        
        dataflow_kwargs=  dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = 'bilinear'
        )
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data,'test'),
            shuffle = False,
            **dataflow_kwargs
        )
        
    @staticmethod
    def load_model(path: Path)-> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    
    def evaluaion(self):
        self.model = self.load_model(self.config.path_to_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        # self.score = {'loss': score[0],'accuracy':score[1]}
        self.save_score()
        
    def save_score(self):
        print("Saving score now...")
        print("Score:", self.score)
        score = {'loss': self.score[0],'accuracy':self.score[1]}
        save_json(path=Path('scores.json'), data=score)

        
   
    def log_into_mlflow(self):
        dagshub.init(
            repo_owner='bhavin2004',
            repo_name='Chest-Cancer-CNN-Classifier-with-MLFLOW-and-DVC',
            mlflow=True
        )

        mlflow.set_tracking_uri("https://dagshub.com/bhavin2004/Chest-Cancer-CNN-Classifier-with-MLFLOW-and-DVC.mlflow")

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)

            mlflow.log_metrics({
                'loss': self.score[0],
                'accuracy': self.score[1]
            })

            # model_path = "artifacts/training/model.h5"
            # # self.model.save(model_path,overwrite=True)  # Save model locally as .h5

            # mlflow.log_artifact(model_path)  # Log it as a generic artifact
