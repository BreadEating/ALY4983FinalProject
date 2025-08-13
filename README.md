# ALY4983FinalProject
## Purpose
We built a simple end-to-end MLOps pipeline for credit default prediction. We downloaded datasets from Default of Credit Card Clients Dataset. Data is versioned with Git + DVC + S3. A small pipeline ingests and validates the dataset, trains a baseline model, and evaluates it. We package a tiny FastAPI service as a Docker image and deploy it to AWS SageMaker. GitHub Actions runs the pipeline on push to main, builds and pushes a Docker image to ECR, and updates the endpoint. CloudWatch shows request metrics, latency, and CPU, with basic alarms. 
## Structure
```
project-root/
├─ data/
│  ├─ raw/
│  │  └─ UCI_Credit_Card.csv.dvc   # DVC pointer
│  └─ staged/                      # generated
├─ src/
│  ├─ data_ingest.py
│  ├─ data_validation.py
│  ├─ train_model.py
│  └─ evaluate.py
├─ artifacts/                      # generated
│  ├─ model.joblib
│  └─ feature_columns.json
├─ inference/
│  └─ predict.py                   # FastAPI app
├─ Dockerfile
├─ entrypoint.sh                   # makes `serve` work on SageMaker
├─ dvc.yaml
├─ dvc.lock
├─ requirements.txt                # runtime deps
├─ .github/workflows/cicd.yml
├─ .dvc/config                     # DVC remote URL
└─ README.md
```
## Prerequisites
* Python 3.10
* AWS CLI configured or role-based access
* Docker Desktop
* A SageMaker execution role ARN
## Setup
```
git clone https://github.com/BreadEating/ALY4983FinalProject.git
```
Setup virtual environment (Optional)
```
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install dvc[s3]
```
DVC remote
```
dvc remote add -d origin s3://<your-bucket>/<prefix>
git add .dvc/config && git commit -m "dvc remote"
```
Upload dataset
```
dvc add data/raw/UCI_Credit_Card.csv
git add data/raw/UCI_Credit_Card.csv.dvc data/raw/.gitignore
git commit -m "track dataset with DVC"
dvc push
```
Reproduce pipeline
```
dvc repro
```
Run API locally (Optional)
```
docker buildx build --platform linux/amd64 --output=type=docker \
  --attest type=provenance,disabled=true \
  -t credit-inference:amd64 .

docker run --rm --platform linux/amd64 -p 8080:8080 credit-inference:amd64 serve

curl -s http://localhost:8080/ping
curl -s -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"features":{"LIMIT_BAL":200000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":30}}'
```
Deploy to AWS
```
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGISTRY=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
ECR_REPO=credit-inference
IMAGE_TAG=v1
IMAGE_URI=${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}
ROLE_ARN=<your-sagemaker-execution-role-arn>
ENDPOINT_NAME=credit-inference-ep
INSTANCE_TYPE=ml.t2.medium
INITIAL_INSTANCE_COUNT=1
```
ECR repo check
```
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION"
```
Push image as Docker v2 manifest
```
aws ecr get-login-password --region "$AWS_REGION" \
| skopeo login --username AWS --password-stdin "$REGISTRY"

skopeo copy --format v2s2 docker-daemon:credit-inference:amd64 docker://"$IMAGE_URI"
```
Create model + endpoint config + endpoint
```
aws sagemaker create-model \
  --model-name "$MODEL_NAME" \
  --primary-container Image="$IMAGE_URI" \
  --execution-role-arn "$ROLE_ARN" \
  --region "$AWS_REGION"

aws sagemaker create-endpoint-config \
  --endpoint-config-name "$CFG_NAME" \
  --production-variants "[{\"VariantName\":\"AllTraffic\",\"ModelName\":\"$MODEL_NAME\",\"InitialInstanceCount\":${INITIAL_INSTANCE_COUNT},\"InstanceType\":\"$INSTANCE_TYPE\"}]"

aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT_NAME" --region "$AWS_REGION" >/dev/null 2>&1 \
  && aws sagemaker update-endpoint --endpoint-name "$ENDPOINT_NAME" --endpoint-config-name "$CFG_NAME" --region "$AWS_REGION" \
  || aws sagemaker create-endpoint --endpoint-name "$ENDPOINT_NAME" --endpoint-config-name "$CFG_NAME" --region "$AWS_REGION"
```
## CI/CD
Workflow file: .github/workflows/cicd.yml
On push to main the job will:
* Configure AWS credentials (from repo Secrets)
* dvc pull (fetch data from S3)
* *vc repro (build artifacts)
* Build Docker, push to ECR
* Create new endpoint config and update the SageMaker endpoint

Set these repo Secrets (Settings → Secrets and variables → Actions):
* AWS_ACCESS_KEY_ID
* AWS_SECRET_ACCESS_KEY
* AWS_REGION (e.g., us-east-1)
* AWS_ACCOUNT_ID (12-digit)
* ECR_REPO (e.g., credit-inference)
* ENDPOINT_NAME (e.g., credit-inference-ep)
* ROLE_ARN (SageMaker execution role)
Re-run a pipeline: push a commit or click Re-run in Actions.

## Dataset source
https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/data