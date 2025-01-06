import json
import base64
import boto3
runtime = boto3.client('runtime.sagemaker') # According to https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2025-01-06-14-08-27-089' 

def lambda_handler(event, context):
    event = event['body']
    # Decode the image data
    image = base64.b64decode(event['image_data'])

    # Make a prediction:
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT,
                                       ContentType='image/png',
                                       Body=image)

    # We return the data back to the Step Function    
    inferences = response['Body'].read().decode('utf-8')
    return {
        'statusCode': 200,
            'body': {
                "inferences": inferences
        }
    }