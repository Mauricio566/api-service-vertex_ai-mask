import requests
import base64


ruta_imagen = "tapabocas2.jpg"

#  
with open(ruta_imagen, "rb") as image_file:
    # We encoded it to base64 and converted it to text (utf-8)
    img_b64 = base64.b64encode(image_file.read()).decode('utf-8')

payload = {
    "instances": [
        {
            "b64": img_b64
        }
    ]
}


url = "http://localhost:8081/inference"

print("sending image to the api...")
respuesta = requests.post(url, json=payload)


if respuesta.status_code == 200:
    print("¡Success! Prediction received:")
    print(respuesta.json())
else:
    print(f"Error {respuesta.status_code}: {respuesta.text}")