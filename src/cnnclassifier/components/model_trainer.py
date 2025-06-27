# 🔴 nothing that touches TensorFlow should be imported above this block
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"                 # hide INFO/WARN
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"  # kill XLA logs

import tensorflow as tf
tf.get_logger().setLevel('ERROR')                        # silence Python logger

import urllib.request as request
from src.cnnclassifier.config.configuration import ConfigurationManager
from src.cnnclassifier.entities.config_entity import ModelTrainerConfig
from pathlib import Path
import math
from tensorflow.keras.applications.vgg16 import preprocess_input



class Training:
    def __init__(self,config: ModelTrainerConfig=ConfigurationManager().get_training_config()) -> None:
        self.config = config
        
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_generator(self):
        
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
            directory=os.path.join(self.config.training_data,'valid'),
            shuffle = False,
            **dataflow_kwargs
        )
        
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator
        
        self.train_generator= train_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data,"train"),
            shuffle=True,
            **dataflow_kwargs
        )
        
    @staticmethod
    def save_model(path: Path,model : tf.keras.Model):
        model.save(path,save_format='keras')
        
        
    def train(self):
        self.get_base_model()
        self.train_valid_generator()
        self.steps_per_epochs = math.ceil(self.train_generator.samples / self.train_generator.batch_size)
        self.validation_steps = math.ceil(self.valid_generator.samples / self.valid_generator.batch_size)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='artifacts/training/model.weights.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
        
        callbacks=[model_checkpoint_callback]
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch = self.steps_per_epochs,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callbacks,
            verbose=2
        )    
        
        self.save_model(
            path=self.config.training_model_path,
            model=self.model
            )
    