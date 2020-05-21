from boto3.s3.transfer import S3Transfer
import boto3

ACCESS_KEY = 'AKIAXRPUYGEFNBNRXE6G'
SECRET_KEY = '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'
BUCKET_NAME = 'smiledatabase'


def upload_file(filepath):
    print('start uploding')
    client = boto3.client('s3', aws_access_key_id=ACCESS_KEY,aws_secret_access_key=SECRET_KEY)
    transfer = S3Transfer(client)
    transfer.upload_file(filepath, BUCKET_NAME, filepath)
    print('end uploading')