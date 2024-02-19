from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Models config
    """
    model_name: str = 'LinearRegression'