import json
import logging
import pickle
import tarfile
import os
import sys

import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def load_csv_dataset(directory, file_name):
    '''
    Load CSV dataset
    '''

    input_csv_path = os.path.join(directory, file_name)
    
    logger.info('Loading dataset at: %s', input_csv_path) 
    dataset_df = pd.read_csv(input_csv_path)
    
    dataset_np = dataset_df.to_numpy()
    
    X = dataset_np[:, 1:]
    y = dataset_np[:, 0]
    
    return X, y


if __name__ == '__main__':
    logger.info('Starting evaluation.')
    
    model_dir = '/opt/ml/processing/model'
    logger.info('List dir %s -> %s', model_dir, os.listdir(model_dir))
    
    model_path = os.path.join(model_dir, 'model.tar.gz')
    with tarfile.open(model_path) as tar:
        tar.extractall(path='.')
        
    logger.info('Loading storred model')
    model = pickle.load(open('model.pickle', 'rb'))

    train_data_dir = '/opt/ml/processing/test'
    X_test, y_test = load_csv_dataset(train_data_dir, 'test.csv')
    
    y_pred = model.predict(X_test)
    test_accuracy_score = accuracy_score(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    report_dict = {
        'binary_classification_metrics': {
            'accuracy': {
                'value': test_accuracy_score
            },
            'confusion_matrix' : {
                '0' : {
                    '0' : int(tn),
                    '1' : int(fn)
                },
                '1' : {
                    '0' : int(fp),
                    '1' : int(tp)
                }
            }
        },
    }

    output_dir = '/opt/ml/processing/evaluation'
    os.makedirs(output_dir, exist_ok=True)

    evaluation_path = f'{output_dir}/evaluation.json'
    with open(evaluation_path, 'w') as f:
        f.write(json.dumps(report_dict))