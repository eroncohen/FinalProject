from boto3.s3.transfer import S3Transfer
import boto3

ACCESS_KEY = 'AKIASCRPHBWZX3ZR67HY'
SECRET_KEY = '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'
BUCKET_NAME = 'smiledatabase2'


def upload_file(file_path):
    print('start uploading')
    client = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    transfer = S3Transfer(client)
    transfer.upload_file(file_path, BUCKET_NAME, file_path)
    print('end uploading')