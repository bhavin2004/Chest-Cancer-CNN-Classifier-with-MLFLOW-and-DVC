from setuptools import setup,find_packages
from typing import List
project_name = 'chest_cancer_cnn_classifier'

def get_requirements(file_path:str)->List[str]:
    with open(file_path,'r') as f:
        requirements=[]
        for line in f.readlines():
            requirements.append(line.replace('\n',''))
        if '-e .' in requirements:
            requirements.remove('-e .')
        return requirements


setup(
    name=project_name,
    version='1.0',
    author="Bhavin Karangia",
    author_email='Karangiabhavin2004@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
)