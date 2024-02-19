import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting the data from the data_path.
    """
    def __init__(self, data_path):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path
    
    def get_data(self):
        """
        return the data at data_path adress.
        """
        logging.info(f"Ingesting info from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path.

    Args:
        data_path: path to the data
    
    Returns:
        pd.DataFrame: the ingested data

    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'error while ingesting the data: {e}')
        raise e
