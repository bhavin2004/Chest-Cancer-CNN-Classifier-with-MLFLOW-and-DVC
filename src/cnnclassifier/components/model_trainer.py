import os
import urllib.request as request
from src.cnnclassifier.config.configuration import ConfigurationManager
from src.cnnclassifier.entities.config_entity import ModelTrainerConfig
import tensorflow as tf
from pathlib import Path


class Training:
    def __init__(self,config: ModelTrainerConfig=ConfigurationManager().get_training_config()) -> None:
        self.config = config
        
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_generator(self):
        
        datagenerator_kwargs = dict(
            rescale = 1./255,
            # validation_split = 0.2
        )
        
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
        model.save(path)
        
        
    def train(self):
        self.get_base_model()
        self.train_valid_generator()
        # self.steps_per_epochs = self.train_generator.samples // self.train_generator.batch_size
        # self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='artifacts/training/model.weights.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
        
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=0,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
            start_from_epoch=0,
        )
        
        callbacks=[early_stopping_callback,model_checkpoint_callback]
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            # steps_per_epoch = self.steps_per_epochs,
            # validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callbacks,
            verbose=2
        )    
        
        self.save_model(
            path=self.config.training_model_path,
            model=self.model
            )
    