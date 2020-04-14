"""
Configuration for the env path and default parameters
"""

import os

BASE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
DATA_PATH = os.path.join(BASE_PATH, 'data')
MISC_PATH = os.path.join(BASE_PATH, 'misc')
ARTIFACTS_PATH = os.path.join(BASE_PATH, 'artifacts')
DB_FILE = os.path.join(MISC_PATH, 'db_config.yaml')
SCHEMA_FILE = os.path.join(MISC_PATH, 'schemas.yaml')

SEED = 1337
MIN_TRAINING_DATA = 1000
HOLDOUT_FILE = 'holdout_id.txt'
