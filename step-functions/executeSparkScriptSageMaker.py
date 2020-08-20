import json
from sagemaker.processing import ScriptProcessor, ProcessingInput
from time import gmtime, strftime
import os 
import boto3
import urllib.request
import sagemaker


def lambda_handler(event, context):
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())    # Variables de ambiente
    local_filename, headers = urllib.request.urlretrieve('https://s3-us-west-2.amazonaws.com/sparkml-mleap/data/abalone/abalone.csv')
    os.rename(local_filename,'/tmp/abalone.csv')
    
    bucket = os.environ['bucket']
    role = os.environ['role']
    sagemaker_session = sagemaker.Session()
    spark_repository_uri = os.environ['spark_repository_uri']
    # Prefix constantes
    prefix = 'sagemaker/spark-preprocess-demo/' + timestamp_prefix 
    input_prefix = prefix + '/input/raw/abalone'
    input_preprocessed_prefix = prefix + '/input/preprocessed/abalone'
    mleap_model_prefix = prefix + '/mleap-model'
    # Store the value of the execution timestamp
    client = boto3.client('s3')
    client.put_object(Body=timestamp_prefix.encode('ascii'),
                        Bucket=bucket,
                        Key='execution.txt')
    # Upload data so it's present for training and inference
    
    print(sagemaker_session.upload_data(path='/tmp/abalone.csv', bucket=bucket, key_prefix=input_prefix))
    
    spark_processor = ScriptProcessor(base_job_name='spark-preprocessor',
                                  image_uri=spark_repository_uri,
                                  command=['/opt/program/submit'],
                                  role=role,
                                  instance_count=2,
                                  instance_type='ml.r5.xlarge',
                                  max_runtime_in_seconds=1200,
                                  env={'mode': 'python'})

    spark_processor.run(code=f's3://{bucket}/sparkdemo/preprocess.py',
                    arguments=['s3_input_bucket', bucket,
                              's3_input_key_prefix', input_prefix,
                              's3_output_bucket', bucket,
                              's3_output_key_prefix', input_preprocessed_prefix,
                              's3_model_bucket', bucket,
                              's3_mleap_model_prefix', mleap_model_prefix],
                    logs=True)
    
    event['s3_output_path'] = f's3://{bucket}/sagemaker/spark-preprocess-demo/{timestamp_prefix}/xgboost_model'
    event['train_data'] = f's3://{bucket}/sagemaker/spark-preprocess-demo/{timestamp_prefix}/input/preprocessed/abalone/train/part'
    event['validation_data'] = f's3://{bucket}/sagemaker/spark-preprocess-demo/{timestamp_prefix}/input/preprocessed/abalone/validation/part'
    event['training_job'] = f'{timestamp_prefix}-job' 
    return event
