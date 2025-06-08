import os
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_name = 'cnnclassifier'
list_of_files=[
    '.github/workflows/.gitkeep',
    'src/__init__.py',
    f'src/{project_name}/__init__.py',
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f'src/{project_name}/config/__init__.py',
    f'src/{project_name}/config/configuration.py',
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/entities/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    'config/config.yaml',
    'dvc.yaml',
    'params.yaml',
    'requirements.txt',
    'setup.py',
    'main.py',
    "research/trails.ipynb",
    # 'README.md',
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Created directory: {file_dir} for file: {file_name}")
    
    if (not os.path.exists(file_path)):
        with open(file_path, 'w') as file:
            pass  # Create an empty file
        logging.info(f"Created file: {file_path}")
    
    else:
        logging.info(f"File already exists: {file_path}")