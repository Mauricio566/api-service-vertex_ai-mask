End To End image clasificacion GCP and vertex AI

Pipeline de Machine Learning para detección de uso de tapabocas utilizando visión por computadora y MLOps en Google Cloud.
este proyecto es con fines practicos, lo desarrollé para entender el flujo completo de un modelo en producción.
quiero agregar que el deploy automático en el pipeline fue omitido por tiempo y enfoque en la parte de entrenamiento y validacion.

Tecnologias

Tensorflow/keras
Scikit-learn
Google Cloud Platform (Vertex AI, Cloud Storage, bigquery)
Kubeflow Pipelines (DSL)
Python
docker

¿Qué hace este proyecto?

Entrenamos un modelo de clasificacion enfocado en 2 clases (con mascarilla y sin mascarilla) comenzamos haciendo uso de un modelo 
previamente entrenado llamado MobileNetV2, entrenamos validamos y guardamos pesos en un bucket de storage.
concluida esta parte se procede a servir el modelo como api en un endpoint de vertex
para esto se utiliza fastapi donde se carga el modelo y se expone un endpoint para recibir imagenes y retornar predicciones

posteriormente el modelo exportado (model.export) es desplegado en vertex ai creando un endpoint para inferencia en la nube
una vez desplegado el modelo se pueden hacer predicciones reales consumiendo el endpoint desde la api
adicionalmente se guarda una version del modelo en formato .keras en un bucket de storage para futuros reentrenamientos

finalmente se construye un pipeline donde se carga el modelo desde el bucket, se reentrena con nuevos datos y se evalua usando metricas para decidir si el modelo debe ser
desplegado o no.

nota: los modelos no se incluyen en el repositorio por su tamaño, se encuentran almacenados externamente.


