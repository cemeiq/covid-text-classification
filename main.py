
import os
import argparse
import json
import joblib
import pickle

from pathlib import Path
import logging
import argparse
from src.scripts.modeling import cnn
import data

def startproject(project_name=None):
    '''
    Creates a standard data science project directory. This helps in
    easy team collaboration, rapid prototyping, easy reproducibility and fast iteration. 
    
    The directory structure is by no means a globally recognized standard, but was inspired by
    the folder structure created by the Azure team (https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview)
    and Edward Ma (https://makcedward.github.io/) of OOCL.
    
    PROJECT STRUCTURE:
            ├── data
            │   ├── processed
            │   └── raw
            ├── outputs
            │   ├── models
            ├── src
            │   ├── scripts
            │       ├── ingest
            │       ├── modeling
            │       ├── preparation
            │       ├── test
            ├   ├── notebooks
            DETAILS:
            data: Stores data used for the experiments, including raw and intermediate processed data.
                processed: stores all processed data files after cleaning, analysis, feature creation etc.
                raw: Stores all raw data obtained from databases, file storages, etc.
            outputs:Stores all output files from an experiment.
                models: Stores trained binary model files. This are models saved after training and evaluation for later use.
            src: Stores all source code including scripts and notebook experiments.
                scripts: Stores all code scripts usually in Python/R format. This is usually refactored from the notebooks.
                    modeling: Stores all scripts and code relating to model building, evaluation and saving.
                    preparation: Stores all scripts used for data preparation and cleaning.
                    ingest: Stores all scripts used for reading in data from different sources like databases, web or file storage.
                    test: Stores all test files for code in scripts.
                notebooks: Stores all jupyter notebooks used for experimentation.
    
    Parameters:
    -------------
        project_name: String, Filepath
            Name of filepath of the directory to initialize and create folders.
        
    Returns:
    -------------
        None
    '''

    DESCRIPTION = '''Creates a standard data science project directory. This helps in
                    easy team collaboration, rapid prototyping, easy reproducibility and fast iteration.       
                    The directory structure is by no means a globally recognized standard, but was inspired by
                    the folder structure created by the Azure team (https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview)
                  '''
    name = ''
    if project_name:
        name = project_name
    else:
        parser = argparse.ArgumentParser(prog='project',description=DESCRIPTION)
        parser.add_argument('name', default='data_project', type=str, help='Name of directory to contain folders')
        args = parser.parse_args()
        name = args.name

    print("Creating project {}".format(name))

    base_path = os.path.join(os.getcwd(), name)
    data_path = os.path.join(base_path, 'data')
    output_path = os.path.join(base_path, 'outputs')
    model_path = os.path.join(output_path, 'models')
    src_path = os.path.join(base_path, 'src')
    scripts_path = os.path.join(base_path, 'src', 'scripts')

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.join(data_path, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'processed'), exist_ok=True)

    os.makedirs(os.path.join(output_path), exist_ok=True)
    os.makedirs(os.path.join(model_path), exist_ok=True)

    os.makedirs(os.path.join(src_path), exist_ok=True)
    os.makedirs(os.path.join(src_path, 'notebooks'), exist_ok=True)

    os.makedirs(os.path.join(scripts_path), exist_ok=True)
    os.makedirs(os.path.join(scripts_path, 'ingest'), exist_ok=True)
    os.makedirs(os.path.join(scripts_path, 'preparation'), exist_ok=True)
    os.makedirs(os.path.join(scripts_path, 'modeling'), exist_ok=True)
    os.makedirs(os.path.join(scripts_path, 'test'), exist_ok=True)

    
    #project configuration settings
    json_config = {
        "description":
        "This file holds all confguration settings for the current project",
        "basepath": base_path,
        "datapath": data_path,
        "outputpath": output_path,
        "modelpath": model_path
    }

    #create a readme.txt file to explain the folder structure
    with open(os.path.join(base_path, "README.txt"), 'w') as readme:
        readme.write(DESCRIPTION)

    with open(os.path.join(base_path, "config.txt"), 'w') as configfile:
        json.dump(json_config, configfile)

    print("Project created successfully in {}".format(base_path))


def _get_home_path(filepath):
    if filepath.endswith('src'):
        indx = filepath.index("src")
        path = filepath[0:indx]
        return path
    elif filepath.endswith('src/scripts/ingest'):
        indx = filepath.index("src/scripts/ingest")
        path = filepath[0:indx]
        return path
    elif filepath.endswith('src/scripts/preparation'):
        indx = filepath.index("src/scripts/preparation")
        path = filepath[0:indx]
        return path
    elif filepath.endswith("src/scripts/modeling"):
        indx = filepath.index("src/scripts/modeling")
        path = filepath[0:indx]
        return path
    elif filepath.endswith("src/notebooks"):
        indx = filepath.index("src/notebooks")
        path = filepath[0:indx]
        return path
    else:
        return filepath


def _get_path(dir=None):
    homedir = _get_home_path(os.getcwd())
    config_path = os.path.join(homedir, 'config.txt')

    with open(config_path) as configfile:
        config = json.load(configfile)

    path = config[dir]
    return path
