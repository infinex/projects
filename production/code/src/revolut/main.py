# -*- coding: utf-8 -*-
"""

Code for Training of xgb model
    Key Steps
        | 1) Loading of Tables from sql database
        | 2) Preprocess step to removal of invalid data points and adding of numeric text features
        | 3) Fit and transformation pipeline
        | 4) Saving of transformation pipeline to be used for prediction
        | 5) Fixing holdout set to ensure consistent validation across multiple run
        | 6) Training  fix at a constant epoch

"""
import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from revolut import env
from revolut import transformer_utils
from revolut.cfg import *
from revolut.db import ETL
from revolut.log import _logger
from revolut.preprocessor import DataFramePreprocessor
from revolut.util import save_xgb, load_xgb, print_classification_report
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

__author__ = "yj"
__copyright__ = "yj"
__license__ = "mit"

MODEL_OUTPUT_NAME = 'model'
LABEL_TRANSFORMER_OUTPUT_NAME = 'label_transformer'
FEATURE_TRANSFORMER_OUTPUT_NAME = 'feature_transformer'


def set_seed(seed):
    """
    fix the seed

    Args:
        seed (int): seed

    Returns:

    """
    random.seed(seed)
    np.random.seed(seed)


class Predictor:
    """
    Predictor class for Predicting of a loaded model
    """

    def __init__(self, args):
        self.model_name = args.model_name
        self.model_base_path = Path(env.ARTIFACTS_PATH) / self.model_name
        self.n_threads = args.n_threads

        self.xgb_model_save_path = self.model_base_path / f'{MODEL_OUTPUT_NAME}.xgb'
        self.label_transformer_save_path = self.model_base_path / f'{LABEL_TRANSFORMER_OUTPUT_NAME}.pkl'
        self.feature_transformer_model_save_path = self.model_base_path / f'{FEATURE_TRANSFORMER_OUTPUT_NAME}.pkl'
        self.input_pipeline = transformer_utils.load_transformer(self.feature_transformer_model_save_path)
        self.label_pipeline = transformer_utils.load_transformer(self.label_transformer_save_path)
        self.xgb_bst = load_xgb(self.xgb_model_save_path)

        self.df_preprocessor = DataFramePreprocessor()

    def predict(self, df):
        df = self.df_preprocessor.prepare_features_for_predicting(df)
        df = self.input_pipeline.transform(df[BASE_TRAINING_FEATURES + [COMPLAINT_TEXT]])
        deval = xgb.DMatrix(df)
        result = self.label_pipeline.inverse_transform(self.xgb_bst.predict(deval).astype(int))
        result = dict(zip([PRODUCT_ID, MAIN_PRODUCT, SUB_PRODUCT], [x.split('**') for x in result][0]))
        return result


class Trainer:
    """
    trainer class for training an xgb model
    """

    def __init__(self, args):
        self.num_rounds = args.num_rounds
        self.model_name = args.model_name
        self.n_threads = args.n_threads
        self.seed = args.seed

        self.model_base_path = Path(env.ARTIFACTS_PATH) / self.model_name
        # transformation
        self.label_transformer_save_path = self.model_base_path / f'{LABEL_TRANSFORMER_OUTPUT_NAME}.pkl'
        self.feature_transformer_model_save_path = self.model_base_path / f'{FEATURE_TRANSFORMER_OUTPUT_NAME}.pkl'

        # tfidf
        self.max_df = args.max_df
        self.min_df = args.min_df
        self.max_features = args.max_features
        self.lowercase = args.lowercase
        self.analyzer = args.analyzer
        self.use_idf = args.use_idf
        self.smooth_idf = args.smooth_idf
        self.sublinear_tf = args.sublinear_tf
        self.stop_words = args.stop_words

        # xgb model
        self.max_depth = args.max_depth
        self.eta = args.eta
        self.subsample = args.subsample
        self.colsample_bytree = args.colsample_bytree
        self.gamma = args.gamma
        self.xgb_lambda = args.xgb_lambda

        self.model_base_path = Path(env.ARTIFACTS_PATH) / self.model_name
        self.xgb_model_save_path = self.model_base_path / f'{MODEL_OUTPUT_NAME}.xgb'

        self.df_preprocessor = DataFramePreprocessor()
        self.input_pipeline = self.create_input_pipeline()
        self.label_pipeline = self.create_label_pipeline()
        set_seed(self.seed)

    def train(self):
        """
        the start of the training function
        """
        # Load Data
        try:
            etl = ETL(env.DB_FILE, env.SCHEMA_FILE)
            complaints_users = etl.load_query(SQL_QUERY_STRING)

            # Preprocess
            df = self.df_preprocessor.prepare_features_for_training(complaints_users, drop_invalid_rows=True)
            train_df, holdout_df = self.load_or_create_train_holdout_set(df)

            # fit and transform Label
            _logger.info('Preparing Label Fitting and Transformation')
            train_y = self.label_pipeline.fit_transform(train_df[LABEL])
            holdout_y = self.label_pipeline.transform(holdout_df[LABEL])
            label_class = self.label_pipeline.classes_
            num_class = len(label_class)

            # # Fit and Transform Text Features
            _logger.info('Preparing Text Features Fitting and Transformation')
            train_x = self.input_pipeline.fit_transform(train_df[BASE_TRAINING_FEATURES + [COMPLAINT_TEXT]])
            holdout_x = self.input_pipeline.transform(holdout_df[BASE_TRAINING_FEATURES + [COMPLAINT_TEXT]])

            # Saving transformer
            transformer_utils.save_transformer(self.label_pipeline, self.label_transformer_save_path)
            transformer_utils.save_transformer(self.input_pipeline, self.feature_transformer_model_save_path)

            # Start Training
            bst = self.train_xgb(train_x, train_y, holdout_x, holdout_y, num_class)
            self.evaluate_xgb(bst, train_x, train_y, label_class, 'Training')
            self.evaluate_xgb(bst, holdout_x, holdout_y, label_class, 'Evaluation')
            save_xgb(bst, self.xgb_model_save_path)

            _logger.info('Training of xgb model completed')
        except Exception as e:
            _logger.error('Unknown exception occurs during training', e)
            raise

    def train_xgb(self, train_x, train_y, holdout_x, holdout_y, num_class):
        """
        Training of xgboost model

        Args:
            train_x: input data x
            train_y: input data y
            holdout_x: holdout data x
            holdout_y: holdout data y
            num_class: number of input class

        Returns:
            xgb.Booster: bst
        """
        _logger.info('Start Training')
        _logger.info(f'Training Size {train_x.shape[0]}')
        _logger.info(f'Holdout Size {holdout_x.shape[0]}')
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dholdout = xgb.DMatrix(holdout_x, label=holdout_y)
        xgb_params = self.get_xgb_parameters(num_class)
        bst = xgb.train(xgb_params, dtrain, self.num_rounds, [(dtrain, 'train'), (dholdout, 'eval')])

        return bst

    def evaluate_xgb(self, bst, x, y, label_class, description):
        """
        Model Evaluation Step

        Args:
            description (string): Training or Evaluation
            label_class: label for the class
            bst (xgb.Booster): xgb booster
            x (np.array): train_x
            y (np.array): train_y

        """
        _logger.info('Start Evaluation')
        deval = xgb.DMatrix(x, label=y)
        prediction = bst.predict(deval)
        print_classification_report(y, prediction, label_class, description)

        return bst

    def create_input_pipeline(self):
        """
        create an scikit-learn features preprocessing pipeline
        """
        base_features_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])

        text_transformer = Pipeline(steps=[
            ('tfidf',
             TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, max_features=self.max_features,
                             lowercase=self.lowercase, analyzer=self.analyzer, use_idf=self.use_idf,
                             smooth_idf=self.smooth_idf, sublinear_tf=self.sublinear_tf, stop_words=self.stop_words)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[('base', base_features_transformer, BASE_TRAINING_FEATURES),
                          ('text', text_transformer, COMPLAINT_TEXT)]
            , sparse_threshold=1.0)

        return preprocessor

    def create_label_pipeline(self):
        """
        create an scikit-learn label preprocessing pipeline
        Returns:

        """
        preprocessor = LabelEncoder()
        return preprocessor

    def load_or_create_train_holdout_set(self, df):
        """
        create an fixed holdout set for model evaluation

        Args:
            df: pd.DataFrame

        Returns:
            pd.DataFrame: te_df
            pd.DataFrame: tt_df
        """
        _logger.info('Loading of holdout set')
        holdout_complaint_id = self.load_or_get_holdout_id(df)
        tr_df = df[~df[COMPLAINT_ID].isin(holdout_complaint_id.tolist())]
        te_df = df[df[COMPLAINT_ID].isin(holdout_complaint_id.tolist())]

        return tr_df, te_df

    @staticmethod
    def load_or_get_holdout_id(df, test_size=0.2):
        """
        get  the complaint_id of the holdout id either from loading from file or creating a new hold out set

        Args:
            df (pd.DataFrame): input df
            test_size (float): size of split for test

        Returns:
            np.array: holdout complain_id
            """
        holdout_path = Path(env.DATA_PATH) / env.HOLDOUT_FILE
        if not Path(holdout_path).exists():
            _, holdout_complaint_id = train_test_split(df[COMPLAINT_ID].values, test_size=test_size,
                                                       random_state=env.SEED,
                                                       stratify=df[LABEL].values)

            with open(holdout_path, 'w') as f:
                np.savetxt(f, holdout_complaint_id, fmt='%i')
        else:
            with open(holdout_path, 'r') as f:
                holdout_complaint_id = np.loadtxt(f)

        return holdout_complaint_id

    def get_xgb_parameters(self, num_class):
        """
        get the parameter use for trainng xgboost

        Args:
            num_class (int):  num of class for prediction

        """
        param = {
            'max_depth': self.max_depth,
            'eta': self.eta,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'gamma': self.gamma,
            'lambda': self.xgb_lambda,
            'objective': 'multi:softmax',
            'num_class': num_class,
            'nthread': self.n_threads,
            'eval_metric': ['mlogloss', 'merror']
        }
        return param


def get_args_parser():
    """
    Define the argument parser

    Returns:
        argparse.Namespace: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description='Start a Model Training or Prediction')
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)

    subparser = parser.add_subparsers(dest='command')

    train = subparser.add_parser('train', help='model training')
    train.add_argument("-n", "--num_rounds", default=30, help='number of xgb rounds')
    train.add_argument("-m", "--model_name", default='default', help='name of model to save')
    train.add_argument("--seed", default=env.SEED, help='seed number')
    train.add_argument("--n_threads", default=4, help='No. of threads')

    # tfidf features
    train.add_argument("--max_df", default=0.95, help='max_df')
    train.add_argument("--min_df", default=10, help='min_df')
    train.add_argument("--max_features", default=20000, help='max_features')
    train.add_argument("--lowercase", default=True, help='lowercase')
    train.add_argument("--analyzer", default='word', help='analyzer')
    train.add_argument("--use_idf", default=True, help='use_idf')
    train.add_argument("--smooth_idf", default=True, help='smooth_idf')
    train.add_argument("--sublinear_tf", default=True, help='sublinear_tf')
    train.add_argument("--stop_words", default='english', help='stop_words')

    # xgb featuresx
    train.add_argument("--max_depth", default=6, help='max_depth')
    train.add_argument("--eta", default=0.1, help='eta')
    train.add_argument("--subsample", default=0.3, help='subsample')
    train.add_argument("--colsample_bytree", default=0.3, help='colsample_bytree')
    train.add_argument("--gamma", default=1, help='gamma')
    train.add_argument("--xgb_lambda", default=2, help='lambda')

    predict = subparser.add_parser('predict', help='predict')
    predict.add_argument("-i", "--input", help='string to predict', required=True)
    predict.add_argument("-m", "--model_name", default='default', help='name of model to load')
    predict.add_argument("--n_threads", default=4, help='No. of threads')

    return parser


def get_run_args():
    """
    Print running arguments

    Returns:
         argparse.Namespace: command line parameters namespace
    """
    args, unknowns = get_args_parser().parse_known_args()
    param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
    print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    return args


def setup_logging(loglevel):
    """
    Setup basic logging

    Args:
        loglevel (int): minimum loglevel for emitting messages

    """

    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main():
    """
    Main entry Point for training and prediction

    Example:
        - python main.py train
        - python main.py predict --input "My input"
    """

    args = get_run_args()
    setup_logging(args.loglevel)
    if args.command == 'train':
        _logger.debug("Starting Training Process")
        Trainer(args).train()
    elif args.command == 'predict':
        input_df = pd.DataFrame([args.input], columns=[COMPLAINT_TEXT])
        result = Predictor(args).predict(input_df)
        print(result)
    else:
        raise ValueError('Unknown command : %s ' % args.command)


if __name__ == "__main__":
    main()
