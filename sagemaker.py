import boto3, json, os
rt = boto3.client("sagemaker-runtime", region_name="us-east-1")
payload = {"features":{"LIMIT_BAL":200000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":30}}
resp = rt.invoke_endpoint(
    EndpointName="credit-inference-ep",
    ContentType="application/json",
    Body=json.dumps(payload),
)
print(resp["Body"].read().decode("utf-8"))