import sagemaker
import boto3
import json
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel
from sagemaker.sparkml.model import SparkMLModel
import os

s3 = boto3.resource('s3')

def lambda_handler(event, context):
    schema_json = get_schema_json()
    # Get the execution ID
    sess = sagemaker.Session()
    obj = s3.Object('ml-lab-mggaska', 'execution.txt')
    exec_id = obj.get()['Body'].read().decode("utf-8") 
    role = os.environ['role']
    print(exec_id)
    # Build variables
    training_job = f'{exec_id}-job'
    mleap_model_prefix = f'sagemaker/spark-preprocess-demo/{exec_id}/mleap-model'
    # Create models for Pipeline
    xgb_model = sagemaker.estimator.Estimator.attach(training_job).create_model()
    sparkml_data = 's3://{}/{}/{}'.format(os.environ['bucket'], mleap_model_prefix, 'model.tar.gz')
    sparkml_model = SparkMLModel(model_data=sparkml_data, env={'SAGEMAKER_SPARKML_SCHEMA' : schema_json})
    
    # Create Pipeline Model
    model_name = 'inference-pipeline-' + exec_id
    sm_model = PipelineModel(name=model_name, role=role, models=[sparkml_model, xgb_model])
    sm_model.transformer(1,'ml.m4.xlarge')
    event['model_name'] = model_name
    event['timestamp_prefix'] = exec_id
    return event
    
def get_schema_json():
    schema = {
    "input": [
        {
            "name": "sex",
            "type": "string"
        }, 
        {
            "name": "length",
            "type": "double"
        }, 
        {
            "name": "diameter",
            "type": "double"
        }, 
        {
            "name": "height",
            "type": "double"
        }, 
        {
            "name": "whole_weight",
            "type": "double"
        }, 
        {
            "name": "shucked_weight",
            "type": "double"
        },
        {
            "name": "viscera_weight",
            "type": "double"
        }, 
        {
            "name": "shell_weight",
            "type": "double"
        }, 
        ],
        "output": 
            {
                "name": "features",
                "type": "double",
                "struct": "vector"
            }
         }
    schema_json = json.dumps(schema)
    return (schema_json)

    

