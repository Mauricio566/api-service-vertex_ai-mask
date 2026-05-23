import os
import base64
from io import BytesIO 
import cv2
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

def encode_image(img):
    _, img_ary = cv2.imencode('.jpg', img)  
    img_bytes = img_ary.tobytes()
    buffer = BytesIO(img_bytes)
    imgb64 = base64.b64encode(buffer.getvalue()).decode()
    return imgb64

def make_prediction(imgpath, params, project_name, endpoint_region, endpoint_id):
    print("Preparing the image and connecting to the endpoint...")
    endpoint_url = f"{endpoint_region}-aiplatform.googleapis.com"
    client_options = {"api_endpoint": endpoint_url}
    
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options) 

   
    img = cv2.imread(imgpath)
    if img is None:
        print(f"Error: The image could not be found or read in {imgpath}")
        return
        
    imgb64 = encode_image(img)
    imgname = os.path.basename(imgpath)
    imgdict = {
        "b64": imgb64,
        "name": imgname,
    }
    instance = json_format.ParseDict(imgdict, Value())
    instances = [instance]
    parameters = json_format.ParseDict(params, Value()) 

    endpoint = client.endpoint_path(
        project=project_name, location=endpoint_region, endpoint=endpoint_id
    ) 

    print("Sending request to Vertex AI... Awaiting AI response...")
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    ) 

    label = ""

    #print(response.predictions)
    probabilidades = response.predictions[0] #Rompemos el corchete exterior
    mask,without_mask = probabilidades

    if mask > without_mask:
        label = "con tapabocas"

    else:
        label = "sin tapabocas"

    print(label)    

if __name__ == "__main__":
    make_prediction(
        imgpath="mauro.jpg",
        params={"dataset_name": "test_images"},
        project_name="PROJECT_ID",
        endpoint_region="us-central1",
        endpoint_id="1247541627002552320" 
    )