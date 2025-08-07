import boto3
import os
from app.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BUCKET_NAME, AWS_REGION

def get_s3_bucket():
    session = boto3.session.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    s3 = session.resource('s3')
    return s3.Bucket(AWS_BUCKET_NAME), session

def upload_file_to_s3(file_path, s3_key):
    bucket, session = get_s3_bucket()
    try:
        bucket.upload_file(file_path, s3_key)
        url = f"https://{bucket.name}.s3.{session.region_name}.amazonaws.com/{s3_key}"
        print(f"S3 업로드 성공: {url}")
        return url
    except Exception as e:
        print(f"S3 업로드 에러: {e}")
        # raise를 해도 되고, None이나 에러 반환 패턴도 가능
        raise

def download_file_from_s3(s3_key, local_path):
    bucket, _ = get_s3_bucket()
    try:
        bucket.download_file(s3_key, local_path)
        print(f"S3 다운로드 성공: {local_path}")
        return local_path
    except Exception as e:
        print(f"S3 다운로드 에러: {e}")
        raise