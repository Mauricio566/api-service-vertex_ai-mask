from google.cloud import aiplatform
from google.cloud.aiplatform import Endpoint, Model
from google.api_core.exceptions import InvalidArgument
import os


#cambios
PROJECT_NAME = os.getenv("PROJECT_ID")
ENDPOINT_LOCATION = "us-central1"              
ENDPOINT_NAME = "endpoint_detector"
MODEL_NAME = "model_detector"


print("Connecting")
aiplatform.init(project=PROJECT_NAME, location=ENDPOINT_LOCATION)
print("Ready!")

print(f"checking if the endpoint  '{ENDPOINT_NAME}' already exists...")


endpoints = Endpoint.list(filter=f'display_name="{ENDPOINT_NAME}"')

if endpoints:
    print(f"An existing endpoint was found¡!")
    endpoint = endpoints[0]
else:

    print(f"the endpoint '{ENDPOINT_NAME}' It doesn't exist, creating a new one...")
    endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)

print("Datos del Endpoint:")
print(f"\tPublic Name: {endpoint.display_name}")
print(f"\tInternal resource name (ID): {endpoint.resource_name}")



print(f"Searching if the model '{MODEL_NAME}'It already exists in the Model Registry...")


models = Model.list(filter=f'display_name="{MODEL_NAME}"')

if models:
    print("¡An existing model was found!")
    model = models[0]
else:
    print(f"the model '{MODEL_NAME}'It does not exist. Registering a custom container...")
    

    IMAGE_URI = f"us-central1-docker.pkg.dev/{PROJECT_NAME}/custom-container-repo-covid/predict-mask:latest"
    
    model = aiplatform.Model.upload(
        display_name=MODEL_NAME,
        serving_container_image_uri=IMAGE_URI,
        serving_container_predict_route="/inference",  
        serving_container_health_route="/health",
        serving_container_environment_variables={"PORT": "8080"},
        serving_container_ports=[8080],
        sync=True,
    )
   

print("Model Data:")
print(f"\tPublic name: {model.display_name}")
print(f"\tInternal resource name (ID): {model.resource_name}")



print("Starting the model deployment on the Endpoint...")



model.deploy(
    endpoint=endpoint,
    deployed_model_display_name=MODEL_NAME,
    traffic_percentage=100,
    machine_type="n1-standard-2", 
    min_replica_count=1,
    max_replica_count=1,
    sync=True,  
)

print("\n========================================================")
print(f"Model {model.display_name} successfully deployed.")
print(f"endpoint : {endpoint.display_name}")
print("========================================================")