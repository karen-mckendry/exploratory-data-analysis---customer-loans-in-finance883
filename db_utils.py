import yaml
from sqlalchemy import create_engine
import pandas as pd

class RDSDatabaseConnector():

    def __init__(self, credentials):
        self.DATABASE_TYPE = 'postgresql'       
        self.DBAPI = 'psycopg2'      
        self.HOST = credentials['RDS_HOST']
        self.PASSWORD = credentials['RDS_PASSWORD']
        self.USER = credentials['RDS_USER']
        self.DATABASE = credentials['RDS_DATABASE']
        self.PORT = credentials['RDS_PORT']

    def init_engine(self):
        engine = create_engine(f"{self.DATABASE_TYPE}+{self.DBAPI}://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}")
        engine.execution_options(isolation_level='AUTOCOMMIT').connect()
        return engine
         
    def RDS_to_pd(self, engine):
        df = pd.read_sql_table('loan_payments', engine)
        return df


def credentials_to_dict():
    with open ('credentials.yaml','r') as f:
        return yaml.safe_load(f)

def pd_to_csv(df):
    df.to_csv('loan_payments.csv', index = False)

credentials = credentials_to_dict()
rds_connection = RDSDatabaseConnector(credentials)
engine = rds_connection.init_engine()
data = rds_connection.RDS_to_pd(engine)
pd_to_csv(data)



