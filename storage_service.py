from google.cloud import storage
from datetime import datetime
import cv2

BUCKET_NAME = "mlops-imagenes-usuarios-vertex"

def guardar_imagen_en_bucket(imagen, resultado):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
        nombre_archivo = f"{resultado}_{timestamp}.jpg"
        
        img_bgr = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img_bgr)
        imagen_bytes = buffer.tobytes()

        cliente = storage.Client()
        bucket = cliente.bucket(BUCKET_NAME)
        
        blob = bucket.blob(f"{resultado}/{nombre_archivo}")
        blob.upload_from_string(imagen_bytes, content_type="image/jpeg")

        #print(f"Imagen guardada: {nombre_archivo}")
        print(f" Imagen guardada en {resultado}: {nombre_archivo}")
    except Exception as e:
        print(f"Error al guardar imagen: {e}")