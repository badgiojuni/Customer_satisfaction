import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model:
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        trains the model
        args:
            X_train: train set
            y_train: training labels
        returns: None
        """
        pass

class LRModel(Model):
    """
    Linear Regression model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        trains the model
        args:
            X_train: train set
            y_train: training labels
        returns: None
        """
        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info("Training model is completed")
            return reg
        except Exception as e:
            logging.error("error in training model: {}".format(e))
            raise e
