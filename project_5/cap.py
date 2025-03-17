#%% 

### Install necessary packages
!pip install boto3 sagemaker pandas matplotlib tqdm
#%% 
### Import required libraries
import os
import json
import boto3
import sagemaker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
#%% 
### Define AWS S3 bucket and role
role = get_execution_role()
session = sagemaker.Session()
bucket = session.default_bucket()

### Data Preparation
#%% 
def download_and_arrange_data():
    s3_client = boto3.client('s3')
    with open('file_list.json', 'r') as f:
        d = json.load(f)
    
    for k, v in d.items():
        print(f"Downloading Images with {k} objects")
        directory = os.path.join('train_data', k)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for file_path in tqdm(v):
            file_name = os.path.basename(file_path).split('.')[0] + '.jpg'
            s3_client.download_file('aft-vbi-pds', os.path.join('bin-images', file_name),
                                    os.path.join(directory, file_name))

# Uncomment below to run the function
# download_and_arrange_data()

### Data Preprocessing

def preprocess_data():
    # Placeholder for data cleaning, resizing images, normalizing, etc.
    print("Data preprocessing completed.")

# preprocess_data()

### Upload data to AWS S3

def upload_to_s3(local_directory, bucket, s3_path):
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            file_path = os.path.join(root, file)
            s3_client.upload_file(file_path, bucket, os.path.join(s3_path, file))

# upload_to_s3('train_data', bucket, 'train_data')

### Model Training
#%% 
# Define hyperparameters
hyperparameters = {
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.001
}
#%% 
# Create training estimator
image_uri = sagemaker.image_uris.retrieve('pytorch', session.boto_region_name, version='1.10', py_version='py38', instance_type='ml.g4dn.xlarge')
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    hyperparameters=hyperparameters,
    output_path=f's3://{bucket}/model_output/',
    sagemaker_session=session
)
#%% 
# Train model
# estimator.fit({'train': f's3://{bucket}/train_data'})

### Hyperparameter Tuning
#%% 
# Define hyperparameter search space
hyperparameter_ranges = {
    'batch_size': sagemaker.parameter.IntegerParameter(16, 64),
    'learning_rate': sagemaker.parameter.ContinuousParameter(0.0001, 0.01)
}
#%% 
# Hyperparameter tuning job setup (example)
# tuner = sagemaker.tuner.HyperparameterTuner(estimator, objective_metric_name='validation-accuracy', hyperparameter_ranges=hyperparameter_ranges, max_jobs=10, max_parallel_jobs=2)
# tuner.fit({'train': f's3://{bucket}/train_data'})

### Model Deployment

# Deploy model
# predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m5.large')

# Query endpoint (example)
# result = predictor.predict(sample_input)

# Delete endpoint after use
# predictor.delete_endpoint()

### Cost Optimization

# Spot training example
# spot_estimator = Estimator(image_uri=image_uri, role=role, instance_count=1, instance_type='ml.g4dn.xlarge', use_spot_instances=True, max_wait=3600, max_run=1800, sagemaker_session=session)

# spot_estimator.fit({'train': f's3://{bucket}/train_data'})
