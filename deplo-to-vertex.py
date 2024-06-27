from google.cloud import aiplatform

# Set your project ID and model ID
project_id = "machine-learning-427708"
model_id = "random_forest_model.pkl"

# Initialize the Vertex AI client
client = aiplatform.gapic.EndpointServiceClient(client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"})

# Define the deployment parameters
deployed_model = aiplatform.gapic.DeployedModel(
    model=aiplatform.gapic.ModelName(project=project_id, location="us-central1", model=model_id),
    display_name="rf_model",
)

# Deploy the model to an endpoint
endpoint = client.create_endpoint(parent=f"projects/{project_id}/locations/us-central1", endpoint=deployed_model)

print(f"Model deployed to endpoint: {endpoint.name}")
