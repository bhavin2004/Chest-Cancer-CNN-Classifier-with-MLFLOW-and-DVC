from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL : str
    local_data_file : Path
    unzip_dir : Path
    


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir:Path
    base_model_path : Path
    updated_base_model_path : Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weight:str
    params_classes: int
    
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir:Path
    training_model_path : Path
    updated_base_model_path : Path
    training_data: Path
    params_image_size: list
    params_epochs: int
    params_batch_size: int
    params_is_augmentation:bool
    
@dataclass(frozen=True)
class EvaluationConfig:
    path_to_model: Path
    training_data: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int
