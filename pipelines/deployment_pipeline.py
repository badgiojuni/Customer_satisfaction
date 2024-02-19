import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from pipelines.utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Deployment Trigger config"""
    min_accuracy: float = 0.01

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """
    MLflow deployment getter parameters
    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction server
        pipeline_step_name: name of the step that deployed the MLflow prediction server
        running: when this flag is set, the step only returns a running service
        model_name: name of the model which is deployed
    """
    pipeline_name: str
    pipeline_step_name: str
    running: bool = True
    model_name: str

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data 

@step
def deployment_trigger(
    accuracy: float,
    config:DeploymentTriggerConfig
    ):
    """
    Implement a simple model Deployment trigger that looks at the input model accuracy 
    """
    return accuracy >= config.min_accuracy

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """
    get the prediction service started by the deployment pipeline 

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction server
        pipeline_step_name: name of the step that deployed the MLflow prediction server
        running: when this flag is set, the step only returns a running service
        model_name: name of the model which is deployed
    """
    #get the MLflow deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    #fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )
    if not existing_services:
        raise RuntimeError(
            f"No Mlflow deployment service found for pipeline {pipeline_name},"
            f'step {pipeline_step_name} and model {model_name}.'
            f"pipeline for the model {model_name} is currently running."
        )
    return existing_services[0]


@step
def predictor(
    service:MLFlowDeploymentService,
    data: str
) -> np.ndarray:
    pass

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.5,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    mse, rmse, r2_score = evaluate(model, X_test, y_test)   
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline():
    pass