import base64
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


model = tf.saved_model.load("model_mask_covid")

class RequestData(BaseModel):
    instances: list


@app.get("/health")
def health():
    return {"status": "healthy"}   


@app.post("/inference")
def predict(data: RequestData):
    img_b64 = data.instances[0]['b64']
    
    img_bytes = base64.b64decode(img_b64)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
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

    return {
        "predictions": preds.numpy().tolist()
    }

#https://7ac9a3c3223f8ea3-dot-us-central1.notebooks.googleusercontent.com/proxy/8081/  

#uvicorn app:app --host 0.0.0.0 --port 8081

#ml_inference enviroment

