import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras # type: ignore 
from tensorflow.keras.layers import Flatten,Dense,Dropout # type: ignore 
from pathlib import Path
from src.cnnclassifier.config.configuration import ConfigurationManager
from src.cnnclassifier.entities.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self,config: PrepareBaseModelConfig=ConfigurationManager().get_prepare_base_model_config()) -> None:# type: ignore 
        self.config = config
        
    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weight,
            include_top= self.config.params_include_top
        )
        
        self.save_model(path=self.config.base_model_path,model=self.model)

        return self.model
    @staticmethod
    def _prepare_full_model(model,classes,freeze_till,learning_rate,freeze_all=False):
        if freeze_all:
            model.trainable=False
        if (freeze_till is not None) and (freeze_till>0):
            i=0
            while freeze_till:
                layer = model.layers[-i]
                if 'conv' in layer.name:
                    print(layer.name)
                    layer.trainable=True
                    freeze_till-=1
                i+=1
                
        
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)

        drop1 = Dropout(0.2)(x)
        
        prediction = Dense(
            classes,
            activation='softmax'
        )(drop1)
            
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction,
        )
        
        full_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        
        full_model.summary()
        
        
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_till=1,
            learning_rate=self.config.params_learning_rate,
            freeze_all=True
            
        )
        keras.utils.plot_model(
        self.full_model,
        to_file="artifacts/model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True,
    )
        
        self.save_model(path=self.config.updated_base_model_path,model=self.full_model)
        return self.full_model
    @staticmethod
    def save_model(path: Path,model : tf.keras.Model):
        model.save(path)