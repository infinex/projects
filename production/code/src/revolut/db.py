import psycopg2
from collections import Counter
import spacy
import pandas as pd
from revolut.log import _logger
import yaml
import psycopg2 as pg


class DBConnection:
    def __init__(self, db_config_file):
        with open(db_config_file) as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.db_config = config.get("pg")

    def __enter__(self):
        _logger.info("Creating DB connection...")
        self.connection = pg.connect(
            host=self.db_config.get("host"),
            port=int(self.db_config.get("port")),
            dbname=self.db_config.get("dbname"),
            user=self.db_config.get("user")
        )
        _logger.info("Connection created!")
        return self.connection

    def __exit__(self, type, value, traceback):
        _logger.info("Closing the DB connection!")
        self.connection.close()


class ETL:
    def __init__(self, db_config_path, schema_config):
        self.db_config_path = db_config_path
        with open(schema_config) as schema_file:
            self.schema = yaml.load(schema_file, Loader=yaml.FullLoader)

    def load_tables(self, select=None):
        """
        Connect to the database connection to load tables

        Args:
            select (list): schema list

        Returns:
            dict: dict of dataframe
        """
        schema_dfs = {}
        with DBConnection(self.db_config_path) as connection:
            for table in self.schema:
                table_name = table['name']
                if select is None or table_name in select:
                    try:
                        sql = f"select * from {table_name};"
                        data = pd.read_sql_query(sql, connection)
                        schema_dfs[f'{table_name}'] = data
                    except Exception as e:
                        _logger.error(f"Could not read table {table_name}:", e)
                        raise
        return schema_dfs

    def load_query(self, query):
        """
        load a sql query

        Args:
            query: sql select statement

        Returns:
            pd.DataFrame

        """
        with DBConnection(self.db_config_path) as connection:
            try:
                data = pd.read_sql_query(query, connection)
            except Exception as e:
                _logger.err(f"Could not read sql query {query}:", e)
                raise
            return data


