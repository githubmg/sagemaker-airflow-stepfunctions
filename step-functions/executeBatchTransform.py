import json
import os
import sagemaker

def lambda_handler(event, context):
    # Params
    timestamp_prefix = event['timestamp_prefix']
    bucket = os.environ['bucket']
    model_name = event['model_name']
    
    sagemaker_session = sagemaker.Session()
    input_data_path = 's3://{}/{}/{}'.format(bucket, 'batch', 'batch_input_abalone.csv')
    output_data_path = 's3://{}/{}/{}'.format(bucket, 'batch_output/abalone', timestamp_prefix)
    
    
    job_name = 'serial-inference-batch-' + timestamp_prefix
    transformer = sagemaker.transformer.Transformer(
        # This was the model created using PipelineModel and it contains feature processing and XGBoost
        model_name = model_name,
        instance_count = 1,
        instance_type = 'ml.m4.xlarge',
        strategy = 'SingleRecord',
        assemble_with = 'Line',
        output_path = output_data_path,
        base_transform_job_name='serial-inference-batch',
        sagemaker_session=sagemaker_session,
        accept = 'text/csv'
    )
    transformer.transform(data = input_data_path,
                          job_name = job_name,
                          content_type = 'text/csv', 
                          split_type = 'Line')
    return {
        'statusCode': 200,
        'body': json.dumps('OK')
    }
