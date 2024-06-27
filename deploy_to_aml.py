from azureml.core import Workspace, Environment, Model, Webservice
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.authentication import ServicePrincipalAuthentication
import os
import json

# Load the workspace configuration from the config.json
with open('config.json') as f:
    config = json.load(f)

svc_pr = ServicePrincipalAuthentication(
    tenant_id=config['tenant_id'],
    service_principal_id=config['client_id'],
    service_principal_password=config['client_secret']
)

ws = Workspace(
    subscription_id=config['subscription_id'],
    resource_group=config['resource_group'],
    workspace_name=config['workspace_name'],
    auth=svc_pr
)

# Register the models
model = Model.register(ws, model_name="random_forest_model", model_path="random_forest_model.pkl")


# Define the environment
env = Environment.from_conda_specification(name="myenv", file_path="environment.yml")

# Define the inference configuration
inference_config = InferenceConfig(
    entry_script="score.py",
    environment=env
)

# Define the deployment configuration
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Check if the service already exists and delete it if it does
service_name = "iris"
try:
    existing_service = Webservice(ws, service_name)
    if existing_service:
        existing_service.delete()
        print(f"Deleted existing service: {service_name}")
except Exception as e:
    print(f"No existing service named {service_name} found. Proceeding with deployment.")

# Deploy the model
service = Model.deploy(
    ws, 
    service_name, 
    [model], 
    inference_config, 
    aci_config
)

service.wait_for_deployment(show_output=True)
print(service.state)
