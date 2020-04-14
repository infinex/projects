import yaml
from pathlib import Path
import codecs
import pandas as pd
import psycopg2 as pg
import logging

DATA_PATH = "../data"
CONFIG_PATH = "db_config.yaml"
SCHEMA_PATH = "schemas.yaml"

logging.basicConfig(level=logging.DEBUG)


class DBConnection:
    def __init__(self, db_config_file):
        with open(db_config_file) as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.db_config = config.get("pg")

    def __enter__(self):
        logging.info("Creating DB connection...")
        self.connection = pg.connect(
            host=self.db_config.get("host"),
            port=int(self.db_config.get("port")),
            dbname=self.db_config.get("dbname"),
            user=self.db_config.get("user")
        )
        logging.info("Connection created!")
        return self.connection

    def __exit__(self, type, value, traceback):
        logging.info("Closing the DB connection!")
        self.connection.close()


class ETL:
    def __init__(self, data_path, db_config_path, schema_config):
        self.db_config_path = db_config_path
        self.data_path = Path(data_path)
        with open(schema_config) as schema_file:
            self.schema = yaml.load(schema_file, Loader=yaml.FullLoader)

    def create_tables(self):
        with DBConnection(self.db_config_path) as connection:
            cur = connection.cursor()
            for table in self.schema:
                try:
                    name = table.get("name")
                    schema = table.get("schema")
                    ddl = f"""CREATE TABLE IF NOT EXISTS {name} ({schema})"""
                    cur.execute(ddl)
                except Exception as e:
                    logging.error(f"Could not create table {name}:", e)
                    raise
            logging.info('Tables succesfully created in the DB!')

            connection.commit()

    def transform_tables(self):
        for table in self.schema:
            try:
                table_name = table.get("name")
                table_source = self.data_path.joinpath(f"{table_name}.csv")
                table_cols = []
                for i in table.get("columns"):
                    table_cols.append(str.upper(i))
                df = pd.read_csv(table_source)

                df_reorder = df[table_cols]
                df_reorder.to_csv(table_source, index=False)
            except Exception as e:
                logging.error(f"Failed to transform table {table_name}:", e)
                raise

    def load_tables(self):
        with DBConnection(self.db_config_path) as connection:

            cur = connection.cursor()

            for table in self.schema:
                try:
                    table_name = table.get("name")
                    table_source = self.data_path.joinpath(f"{table_name}.csv")
                    with codecs.open(table_source, "r", encoding="utf8") as f:
                        next(f)
                        cur.copy_expert(f"COPY {table_name} FROM STDIN CSV NULL AS ''", f)
                    connection.commit()
                except Exception as e:
                    logging.error(f"Failed to load table {table_name}:", e)
                    raise

        logging.info("Data were sucessfully loaded in the DB :) \n  ***** You are ready to start the Challenge! *****")


if __name__ == "__main__":

    etl = ETL(DATA_PATH, CONFIG_PATH, SCHEMA_PATH)
    etl.create_tables()
    etl.transform_tables()
    etl.load_tables()
