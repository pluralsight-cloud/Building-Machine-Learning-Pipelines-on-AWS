import argparse
import os
import logging
import sys

import pandas as pd
import numpy as np

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def parse_args():
    logger.info('Parsing arguments')
    parser = argparse.ArgumentParser(description='Churn data processing')

    parser.add_argument('--input-data', type=str,
        default='/opt/ml/processing/input/data',
    )
    parser.add_argument('--output-train', type=str,
        default='/opt/ml/processing/output/train',
    )
    parser.add_argument('--output-validation', type=str,
        default='/opt/ml/processing/output/validation',
    )
    parser.add_argument('--output-test', type=str,
        default='/opt/ml/processing/output/test',
    )

    return parser.parse_args()

if __name__ == '__main__':
    logger.info('Starated preprocessing. Args: %s', sys.argv)
    
    args = parse_args()
    
    input_csv_path = os.path.join(args.input_data, 'churn_raw_data.csv')
    logger.info('Reading data from %s', input_csv_path)
    churn_df = pd.read_csv(input_csv_path)
    
    logger.info('Starting preprocessing')
    
    churn_df = churn_df.drop('Phone', axis=1)
    churn_df['Area Code'] = churn_df['Area Code'].astype(object)
    churn_df['Churn?'] = np.where(churn_df['Churn?'] == 'False.', 0, 1)
    churn_df = pd.concat(
        [churn_df['Churn?'], churn_df.drop(['Churn?'], axis=1)], axis=1
    )
    churn_df = pd.get_dummies(churn_df)
    
    logger.info('Splitting data')
    churn_df_shuffled = churn_df.sample(frac=1, random_state=42)
    churn_df_len = len(churn_df_shuffled)
    churn_df_train, churn_df_validate, churn_df_test = np.split(
        churn_df_shuffled, 
        [
            int(0.6 * churn_df_len),
            int(0.8 * churn_df_len)
        ]
    )
    
    churn_df_train.to_csv(
        os.path.join(args.output_train, 'train.csv'),
        header=False,
        index=False
    )
    churn_df_validate.to_csv(
        os.path.join(args.output_validation, 'validation.csv'),
        header=False,
        index=False
    )
    churn_df_test.to_csv(
        os.path.join(args.output_test, 'test.csv'),
        header=False,
        index=False
    )
    