"""
Preprocessing Utils Function
"""

from collections import Counter
from revolut.log import _logger
import pandas as pd
import spacy
from revolut.cfg import *
from revolut import env
import re


class DataFramePreprocessor:
    """
    Preprocessing Task for a preparation of data for training and prediction
    """

    def __init__(self):
        pass

    def prepare_features_for_predicting(self, df):
        """
        prepare numeric features
        Args:
            df: input pd.DataFrame

        Returns:
            pd.DataFrame

        """
        _logger.info('Preparing Model Features for predicting')
        try:
            df = self.generate_numeric_features(df)
        except Exception as e:
            _logger.error(f"Unable to prepare model features for predicting", e)
            raise
        return df

    def prepare_features_for_training(self, df, drop_invalid_rows=True):
        """
        Prepare numeric nlp features, add label features and add drop flags

        Args:
            df (pd.DataFrame):  input DataFrame
            drop_invalid_rows (bool): flags to indicate whether to remove invalid rows

        Returns:
            df (pd.DataFrame):  output DataFrame
        """

        _logger.info('Preparing Model Features for training')
        try:
            df = self.generate_numeric_features(df)
            df = self.generate_label_features(df)
            df = self.generate_drop_valid_features(df, drop_invalid_rows=drop_invalid_rows)

        except Exception as e:
            _logger.error(f"Unable to prepare model features for training", e)
            raise
        return df

    def generate_numeric_features(self, df):
        """
        Generate numeric features

        Args:
            df (pd.DataFrame):  input DataFrame

        Returns:
            df (pd.DataFrame):  output DataFrame

        """
        df = df.assign(**{
            F_COMPLAINT_LEN: lambda x: x[COMPLAINT_TEXT].apply(lambda x: len(x)),
            F_COMPLAINT_CAPS_LEN: lambda x: x[COMPLAINT_TEXT].apply(lambda x: len(re.findall('[A-Z]', x))),
            F_COMPLAINT_NON_ALPHA_LEN: lambda x: x[COMPLAINT_TEXT].apply(lambda x: len(re.findall('^[A-Z]', x))),
            F_COMPLAINT_EXCLAIMENTION_LEN: lambda x: x[COMPLAINT_TEXT].apply(lambda x: len(re.findall('[!]', x))),
            F_COMPLAINT_MASK_LEN: lambda x: x[COMPLAINT_TEXT].apply(lambda x: len(re.findall('(XX)', x))),
            F_COMPLAINT_END_LEN: lambda x: x[COMPLAINT_TEXT].apply(lambda x: len(re.findall('[!?.]', x))),
            F_COMPLAINT_DIGITS_LEN: lambda x: x[COMPLAINT_TEXT].apply(lambda x: len(re.findall('[0-9]', x))),
            F_COMPLAINT_CAPS_LEN_NORM: lambda x: x[F_COMPLAINT_CAPS_LEN] / x[F_COMPLAINT_LEN],
            F_COMPLAINT_NON_ALPHA_LEN_NORM: lambda x: x[F_COMPLAINT_NON_ALPHA_LEN] / x[F_COMPLAINT_LEN],
            F_COMPLAINT_EXCLAIMENTION_LEN_NORM: lambda x: x[F_COMPLAINT_EXCLAIMENTION_LEN] / x[F_COMPLAINT_LEN],
            F_COMPLAINT_MASK_LEN_NORM: lambda x: x[F_COMPLAINT_MASK_LEN] / x[F_COMPLAINT_LEN],
            F_COMPLAINT_END_LEN_NORM: lambda x: x[F_COMPLAINT_END_LEN] / x[F_COMPLAINT_LEN],
            F_COMPLAINT_DIGITS_LEN_NORM: lambda x: x[F_COMPLAINT_DIGITS_LEN] / x[F_COMPLAINT_LEN],
        })

        return df

    def generate_label_features(self, df):
        """
        Add Predict Label to DataFrame

        Args:
            df (pd.DataFrame) : input DataFrame

        Returns:
            df (pd.DataFrame):  output DataFrame

        """
        # add Label as prediction
        df[LABEL] = df.apply(lambda x: f'{x[PRODUCT_ID]}**{x[MAIN_PRODUCT]}**{x[SUB_PRODUCT]}', axis=1)
        return df

    def generate_drop_valid_features(self, df, drop_invalid_rows):
        """
        Add invalid drop flag to remove missing sub product, duplicated records and low product counts item

        Args:
            df (pd.DataFrame): DataFrame with sub product, complain text and product id
            drop_invalid_rows (bool):  Flag to indicate whether to remove invalid

        Returns:
            df (pd.DataFrame):  output DataFrame

        """
        # Finding Sub Product with Missing Value
        bool_sub_product_is_none = pd.isna(df[SUB_PRODUCT])
        # Finding Sub Product with Duplicated records
        bool_duplicate = df.duplicated([COMPLAINT_TEXT], keep=False)
        # Finding Sub Product with lesser count than 1000
        bool_invalid_training = df.groupby(PRODUCT_ID).transform('count')[
                                    MAIN_PRODUCT] < env.MIN_TRAINING_DATA
        # Find rows to be drop
        invalid_rows = bool_invalid_training | bool_duplicate | bool_sub_product_is_none

        # drop rows
        if drop_invalid_rows:
            df = df[~invalid_rows]
        else:
            # add drop to data frame
            df[DROP] = invalid_rows
        return df


class TextProcessor:
    def __init__(self, ):
        self.nlp = spacy.load("en_core_web_sm")

    def lemmatizer(self, text):
        """
        Spacy lemmatizer

        Args:
            text (string): text of string to lemmatize

        Returns:
            list: lemma tokens

        """
        sent = []
        if text is None:
            return sent
        doc = self.nlp(text, disable=["tagger", "parser", 'ner', 'textcat'])
        for word in doc:
            sent.append(word.lemma_)
        return sent

    @staticmethod
    def overlap_dict(l1, l2):
        """
        overlap between two list

        Args:
            l1 (list): list 1
            l2 (list): list 2

        Returns:
            dict: overlap dictionary
        """
        ctr = Counter(l1) & Counter(l2)
        return dict(ctr)
