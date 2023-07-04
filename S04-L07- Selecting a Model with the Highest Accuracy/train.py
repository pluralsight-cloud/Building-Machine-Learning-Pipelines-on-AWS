import argparse
import logging
import os
import sys

import pickle

import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        '--max-depth',
        type=int,
        default=5
    )
    parser.add_argument(
        '--min-samples-split',
        type=int,
        default=2
    )
    
    # Datasets
    parser.add_argument(
        '--train-data',
        type=str,
        default=os.getenv('SM_CHANNEL_TRAIN')
    )
    parser.add_argument(
        "--validation-data",
        type=str,
        default=os.getenv('SM_CHANNEL_VALIDATION')
    )
    
    # Training output
    parser.add_argument(
        '--model-dir',
        type=str,
        default=os.getenv('SM_MODEL_DIR')
    )

    return parser.parse_args()


def load_csv_dataset(directory, file_name):
    """
    Load CSV dataset
    """

    input_csv_path = os.path.join(directory, file_name)
    
    logger.info('Loading dataset at: %s', input_csv_path) 
    dataset_df = pd.read_csv(input_csv_path)
    
    dataset_np = dataset_df.to_numpy()
    
    X = dataset_np[:, 1:]
    y = dataset_np[:, 0]
    
    return X, y


if __name__ == '__main__':
    
    args = parse_args()

    X_train, y_train = load_csv_dataset(args.train_data, 'train.csv')
    X_validation, y_validation = load_csv_dataset(args.validation_data, 'validation.csv')

    logger.info('Training model')

    model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split
    )

    model.fit(X_train, y_train)

    logger.info('Computing metrics')
    logger.info('train_acc: %s', accuracy_score(
        y_train,
        model.predict(X_train)
    ))
    logger.info('val_acc: %s', accuracy_score(
        y_validation,
        model.predict(X_validation)
    ))
    
    
    model_output_path = os.path.join(args.model_dir, "model.pickle")
    
    logger.info('Writing model to %s', model_output_path)
    with open(model_output_path, 'wb') as model_output_file:
        pickle.dump(model, model_output_file)

        
def model_fn(model_dir):
    """
    Load the model for inference
    """
    model_path = os.path.join(model_dir, "model.pickle")
    
    return pickle.load(open(model_path, 'rb'))


def predict_fn(input_data, model):
    """
    Use model for inference on the input data
    """
    return model.predict(input_data)



