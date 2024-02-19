import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    abstracting method defining strategy for evaluating our models
    """

    @abstractmethod
    def calculate_scores(self, y_true:np.array, y_pred: np.array):
        """
        Evalutates the score for a model:
        args:
            y_true: true labels:
            y_pred: predicted labels:
        returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Square Error
    """
    def calculate_scores(self, y_true:np.array, y_pred: np.array):
        try:
            logging.info('Calculating MSE')
            mse = mean_squared_error(y_true, y_pred)
            logging.info('MSE: {}'.format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
            
    
class R2(Evaluation):
    """
    Evaluation strategy that uses R2 score
    """
    def calculate_scores(self, y_true:np.array, y_pred: np.array):
        try:
            logging.info('Calculating R2 score')
            r2 = r2_score(y_true, y_pred)
            logging.info('R2 score: {}'.format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 score: {}".format(e))
            raise e
        

class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Square Error
    """
    def calculate_scores(self, y_true:np.array, y_pred: np.array):
        try:
            logging.info('Calculating RMSE')
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info('RMSE: {}'.format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e