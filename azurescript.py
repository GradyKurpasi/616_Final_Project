import os#, random
import azureml
import shutil
# import urllib.request
# from pathlib import Path
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import AxesGrid
# import cv2
# import urllib3 
# import zipfile


# from azureml.core.model import Model, InferenceConfig
from azureml.core import Workspace, Datastore, Experiment, Run, Environment, ScriptRunConfig 
from azureml.core.compute import ComputeTarget #, AmlCompute, AksCompute
from azureml.core.dataset import Dataset
from azureml.widgets import RunDetails



# from azureml.core.model import Model, InferenceConfig

# from azureml.core.compute import ComputeTarget, AmlCompute, AksCompute, ComputeTarget
# from azureml.train.dnn import PyTorch
# from azureml.widgets import RunDetails

# from azureml.core.webservice import Webservice, AksWebservice, AciWebservice
# from azureml.core.dataset import Dataset
# from azureml.core.resource_configuration import ResourceConfiguration
# from azureml.core.conda_dependencies import CondaDependencies 


az_workspace = None
az_computetarget = None
az_datastore = None
az_experiment = None
az_dataset = None
az_config = None

def ConnectToAzure():
    """
        Connect to Azure workspace, Compute Target, DataStore and Experiement
    """
    
    # Connect to workspace
    # config.json file expected in ./azureml directory
    # config.json can be generated from the azure portal while browsing the workspace
    global az_workspace
    az_workspace = Workspace.from_config()
    print("Workspace:", az_workspace.name)

    # Connect to compute for training
    # compute target must belong to the workspace AND compute targets are limited by the workspace region
    # there may be ability to do cross workspace compute targets in the future
    global az_computetarget
    az_computetarget = ComputeTarget(workspace=az_workspace, name="AzPytrch-NC6")
    print("Compute Target:", az_computetarget.name)

    # Connect to the datastore for the training images
    # datastore must be associated with storage account belonging to workspace
    global az_datastore
    az_datastore = Datastore.get_default(az_workspace)
    print("Datastore:", az_datastore.name)

    # Connect to the experiment
    global az_experiment
    az_experiment = Experiment(workspace=az_workspace, name='ANFIS-PyTorch')
    print("Experiment:",az_experiment.name)

def UploadData():
    """
        Upload all the data to the datastore
        will upload all files and subdirectories in data_path
    """
    global az_datastore
    data_path = "./data/"  
    az_datastore.upload(src_dir=data_path, target_path='data', overwrite=True, show_progress=True)

def CreateAzureDataset():
    """
        create a FileDataset pointing to files in 'data' folder and its subfolders recursively
    """
    global az_datastore
    datastore_paths = [(az_datastore, 'data')]
    print(datastore_paths)
    global az_dataset
    az_dataset = Dataset.File.from_files(path=datastore_paths)

    # Register the dataset in AMLS
    az_dataset.register(workspace=az_workspace,
                name='616_data',
                description='Data for 616 final',
                create_new_version = True)

    # OR you can connect to existing / previous version of the data
    #azure_ds = Dataset.get_by_name(work_space, name='data')

def PrepareAzureScript():
    """
        Create Script Run Config
    """

    # Use an Azure curated environment to create docker container
    curated_env_name = 'AzureML-PyTorch-1.6-GPU'

    pytorch_env = Environment.get(workspace=az_workspace, name=curated_env_name)
    pytorch_env = pytorch_env.clone(new_name='pytorch-1.6-gpu')

    # OR 
    # use build the conda environment used on local machine (from a python terminal) to create docker container
    # build yml file with 'conda env export -n [name of environment] -f [filename.yml]'
    # place yml file in the ./azureml directory

    # pytorch_env = Environment.from_conda_specification(
    #         name='AzurePytorch',
    #         file_path='./.azureml/AzurePytorch.yml'
    #     )

    # arguments can be passed to training script 
    # they have to be parsed in the training script
    #    import argparse
    #    parser = argparse.ArgumentParser()
    #    parser.add_argument("--data-folder", type=str, dest="data_folder", help="data folder mounting point", default="")
    #    parser.add_argument("--num-epochs", type=int, dest="num_epochs", help="Number of epochs", default="")
    #    args = parser.parse_args()
    #    data_path = args.data_folder
    args = [
        '--data-folder', az_dataset.as_named_input('data').as_mount(),
        '--num-epochs', 50000
    ]

    # Script Run Config defines the wrapper for the python scripts and will be used to create the Docker container
    project_folder = "./scripts"  # this refers to local location of scripts, these scripts will be built into docker container
    global az_config
    global az_computetarget
    az_config = ScriptRunConfig(
        source_directory = project_folder, 
        script = 'preprocess.py', 
        compute_target=az_computetarget,
        environment = pytorch_env,
        arguments=args,
    )

def RunAzureScript():
    # Run script
    # This command can take a long time to run the first time
    # Docker container will be built and published and compute cores may have to be started
    # Subsequent runs will be faster
    global az_experiment
    global az_config
    run = az_experiment.submit(az_config)
    # Show the PyTorch estimator widget
    RunDetails(run).show()



