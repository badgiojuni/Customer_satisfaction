import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow

from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate(model:RegressorMixin,
             X_test: pd.DataFrame,
             y_test: pd.DataFrame
 ) ->Tuple[
     Annotated[float, 'MSE'],
     Annotated[float, 'RMSE'],
     Annotated[float, 'R2']
 ] :
    """
    Evaluates the model on ingested data

    Args:
        df: the ingested data
    """
    try:
        predictions = model.predict(X_test)
        
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("mse", mse)
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("rmse", rmse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("r2", r2)

        return mse, rmse, r2

    except Exception as e:
        logging.error('Error in evaluating the model: {}'.format(e))
        raise e