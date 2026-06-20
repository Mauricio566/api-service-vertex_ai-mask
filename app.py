import base64
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

from google.cloud import bigquery
from datetime import datetime

from storage_service import guardar_imagen_en_bucket

app = FastAPI()


model = tf.saved_model.load("model_mask_covid")

bq_client = bigquery.Client()
TABLE_ID = "project-a5d39078-68b8-4a9c-a89.ml_monitoring.predictions"

class RequestData(BaseModel):
    instances: list


@app.get("/health")
def health():
    return {"status": "healthy"}   


@app.post("/inference")
def predict(data: RequestData):
    img_b64 = data.instances[0]['b64']
    img_name = data.instances[0].get('name', 'unknown')
    
    img_bytes = base64.b64decode(img_b64)
    
    img_original = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)

    
    #img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    #img = cv2.resize(img, (224, 224))
    img = cv2.resize(img_original, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    tensor_img = tf.convert_to_tensor(img)

    
    signature_key = list(model.signatures.keys())[0]
    infer = model.signatures[signature_key]
 
    try:
        preds_dict = infer(tensor_img)
    except:
      
        input_name = list(infer.structured_input_signature[1].keys())[0]
        preds_dict = infer(**{input_name: tensor_img})

    
    preds = list(preds_dict.values())[0]
    preds_array = preds.numpy().tolist()[0]
    
    mask, without_mask = preds_array

    if mask > without_mask:
        prediction_label = "con-tapabocas"
        confidence = float(mask)
    else:
        prediction_label = "sin-tapabocas"
        confidence = float(without_mask)

    guardar_imagen_en_bucket(img_original, prediction_label)    

    row = [{
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": prediction_label,
        "confidence": confidence,
        #"image_name": "unknown"
        "image_name": img_name
        
       
    }]

    errors = bq_client.insert_rows_json(TABLE_ID, row)
    if errors:
        print("Error insertando en BigQuery:", errors)
    else:
        print("Registro guardado en BigQuery")

        

    return {
        #"predictions": preds.numpy().tolist()
        "predictions": preds_array,
        "label": prediction_label,
        "confidence": confidence
    }

#https://7ac9a3c3223f8ea3-dot-us-central1.notebooks.googleusercontent.com/proxy/8081/  

#uvicorn app:app --host 0.0.0.0 --port 8081

#ml_inference enviroment

