import os
import subprocess

STORAGE_BUCKET = 'machine_learning_experiment'
STORAGE_PATH = 'tutorial'
DATA_PATH = 'tutorial/bank-additional.csv'
LOCAL_PATH = '/tmp'
PROJECT_ID = 'gravel-technology'

if __name__ == '__main__':
  # download align face from datasets
  subprocess.call([
    'gsutil', '-m','cp',
    '-r',
    # storage path
    os.path.join('gs://', STORAGE_BUCKET, 'faces/align/align'),
    os.path.join('./datasets', 'align'),
  ])