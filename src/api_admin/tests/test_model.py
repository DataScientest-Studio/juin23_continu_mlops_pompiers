from joblib import load
from decouple import config
import boto3
import pytest

s3_client = boto3.client('s3',region_name=config('AWS_S3_REGION'), aws_access_key_id=config('AWS_ADMIN_KEY_ID'), aws_secret_access_key=config('AWS_ADMIN_KEY'))

s3_client.download_file(Bucket=config('AWS_S3_BUCKET_NAME'), Key='lightgbm/r2_lgb.joblib', Filename='models/r2_lgb.joblib')
r2_lgb = load('models/r2_lgb.joblib')

s3_client.download_file(Bucket=config('AWS_S3_BUCKET_NAME'), Key='lightgbm/rmse_lgb.joblib', Filename='models/rmse_lgb.joblib')
rmse_lgb = load('models/rmse_lgb.joblib')

# Vérification de la performance du modèle LGBM
def test_r2():
  assert r2_lgb > 0.5

def test_rmse():
  assert rmse_lgb < 120