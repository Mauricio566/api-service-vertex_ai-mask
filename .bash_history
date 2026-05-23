gsutil cp -r model_mask_covid gs://ml-models-mascarilla/model_mask_covid/
gcloud auth print-access-token
ENDPOINT_ID="5562869679325708288"
PROJECT_ID="511076842936"
wget -O test.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Man_with_face_mask.jpg/800px-Man_with_face_mask.jpg"
wget -O test.jpg "https://images.pexels.com/photos/3902882/pexels-photo-3902882.jpeg"
python3 -c "
import json, base64
with open('tapabocas.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()
with open('input.json', 'w') as f:
    json.dump({'instances': [{'b64': img_b64}]}, f)
print('listo')
"
ENDPOINT_ID="5562869679325708288" PROJECT_ID="511076842936" INPUT_DATA_FILE="input.json"
CLS
cls
curl  -X POST  -H "Authorization: Bearer $(gcloud auth print-access-token)"  -H "Content-Type: application/json"  "https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict"  -d "@${INPUT_DATA_FILE}"
clear
gcloud auth application-default login
 -X POST  -H "Authorization: Bearer $(gcloud auth print-access-token)"  -H "Content-Type: application/json"  "https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict"  -d "@${INPUT_DATA_FILE}"clear
clear
gcloud auth application-default login --no-launch-browser
curl  -X POST  -H "Authorization: Bearer $(gcloud auth print-access-token)"  -H "Content-Type: application/json"  "https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict"  -d "@${INPUT_DATA_FILE}"
conda activate ml2
python prueba.py
python prueba.py
clear
python prueba.py
python prueba.py
conda activate ml2
uvicorn app:app --host 0.0.0.0 --port 8081}
uvicorn app:app --host 0.0.0.0 --port 8081
clear
uvicorn app:app --host 0.0.0.0 --port 8081
uvicorn app:app --host 0.0.0.0 --port 8080
clear
uvicorn app:app --host 0.0.0.0 --port 8081
uvicorn app:app --host 0.0.0.0 --port 8081
clear
uvicorn app:app --host 0.0.0.0 --port 8081
python app.py
python app.py
cls
clear
pip install Flask
clear
python app.py
python app.py
python app.py
clear
python app.py
pip install opencv-python
python app.py
pip uninstall -y numpy opencv-python
pip install numpy==1.26.4 opencv-python
python app.py
clear
python app.py
clear
micromamba activate ml
/opt/micromamba/envs/ml/bin/python
curl http://127.0.0.1:8081
python -c "import tensorflow as tf; print(tf.__version__)"
pip install tensorflow==2.14.0
pip uninstall -y tensorflow keras
pip install tensorflow==2.14.0
clear
python3.10 --version
conda create -n ml python=3.10
micromamba activate ml
python --version
pip list
pip install tensorflow==2.14.0
python -c "import tensorflow as tf; print(tf.__version__)"
pip list
pip uninstall -y numpy ml-dtypes protobuf
pip list
pip install numpy==1.23.5
pip install protobuf==4.25.9
pip install ml-dtypes==0.2.0
pip list
python -c "import tensorflow as tf; import numpy as np; import google.protobuf as p; print(tf.__version__, np.__version__, p.__version__)"
micromamba activate ml
python app.py
clear
pip list
clear
pip install fastapi==0.136.0
clear
python app.py
clear
pip install opencv-python==4.13.0.92
clear
python app.py
pip install --force-reinstall numpy==1.23.5
pip install --force-reinstall ml-dtypes==0.2.0
pip install --force-reinstall protobuf==4.25.9
pip install --force-reinstall tensorflow==2.14.0
python app.py
micromamba create -n ml2 python=3.10 -y
micromamba activate ml2
pip install tensorflow==2.14.0 numpy==1.23.5 ml-dtypes==0.2.0 protobuf==4.25.9
python -c "import tensorflow as tf; print('TF OK')"
pip install fastapi==0.136.0 uvicorn==0.44.0 opencv-python
python app.py
pip list
pip uninstall numpy -y
pip list
pip install numpy==1.23.5 --no-deps
pip list
python app.py
clear
ls
clear
python app.py
uvicorn app:app --host 0.0.0.0 --port 8081
clear
pip install pyngrok --no-deps
python run.py
python run.py
clear
python run.py
pip install pyyaml
python run.py
clear
uvicorn app:app --host 0.0.0.0 --port 8080
uvicorn app:app --host 0.0.0.0 --port 8081
clear
uvicorn app:app --host 0.0.0.0 --port 8080
clear
uvicorn app:app --host 0.0.0.0 --port 8081
conda activate ml2
python prueba.py
python prueba.py
python prueba.py
python prueba.py
conda activate ml_inference
clear
python prueba.py
clear
python prueba.py
python prueba.py
micromamba create -n ml_inference python=3.10 -c conda-forge -y
micromamba activate ml_inference
pip install tensorflow==2.21.0 fastapi uvicorn opencv-python numpy pydantic requests
clear
uvicorn app:app --host 0.0.0.0 --port 8081
clear
uvicorn app:app --host 0.0.0.0 --port 8081
clear
uvicorn app:app --host 0.0.0.0 --port 8081
clear
uvicorn app:app --host 0.0.0.0 --port 8081
conda activate ml2
uvicorn app:app --host 0.0.0.0 --port 8081
clear
uvicorn app:app --host 0.0.0.0 --port 8081
clear
uvicorn app:app --host 0.0.0.0 --port 8081
clear
conda activate ml3
conda activate ml2
clear
python prueba.py
python prueba.py
python prueba.py
python prueba.py
pip list
micromamba env list
conda activate ml_inference
clear
python prueba.py
python prueba.py
pip freeze > requirements.txt
conda activate ml2
clear
uvicorn app:app --host 0.0.0.0 --port 8081)
uvicorn app:app --host 0.0.0.0 --port 808clear)
clear
uvicorn app:app --host 0.0.0.0 --port 808clear
clear
uvicorn app:app --host 0.0.0.0 --port 8081
conda activate ml_inference
uvicorn app:app --host 0.0.0.0 --port 8081
docker ps
clear
python prueba.py
clear
python prueba.py
clear
export PROJECT_ID=$(gcloud config get-value project)
gcloud artifacts repositories create covid-repo     --repository-format=docker     --location=us-central1     --description="Almacen para el contenedor de mascaras"
gcloud auth configure-docker us-central1-docker.pkg.dev -q
export PROJECT_ID=$(gcloud config get-value project)
docker tag mask-api us-central1-docker.pkg.dev/$PROJECT_ID/custom-container-repo-covid/predict-mask:latest
docker push us-central1-docker.pkg.dev/$PROJECT_ID/custom-container-repo-covid/predict-mask:latest
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev -q
docker push us-central1-docker.pkg.dev/project-a5d39078-68b8-4a9c-a89/custom-container-repo-covid/predict-mask:latest
conda activate ml_inference
python deploy.app
python deploy.py
python deploy.py
python deploy.py
clear
python deploy.py
conda activate ml_inference
python --version
docker build -t mask-api .
clear
docker build -t mask-api .
docker run -p 8080:8080 mask-api
docker run -p 8081:8080 mask-api
clear
docker build -t mask-api .
docker run -p 8081:8080 mask-api
docker run -p 8081:8080 mask-api
clear
docker run -p 8081:8080 mask-api
pip show google-cloud-aiplatform
pip install google-cloud-aiplatform --upgrade-strategy=only-if-needed --user
clear
python predict.py 
conda activate ml_inference
clear
python predict.py
python predict.py
python predict.py
python predict.py
python predict.py
clear
python predict.py
python shutdown.py
