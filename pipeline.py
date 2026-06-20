from kfp import dsl

@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
    "google-cloud-bigquery",
    "pandas",
    "db-dtypes",
    "opencv-python-headless",
    "google-cloud-storage" 
        
    ]
    
)

def cargar_datos( 
    data_path: dsl.Output[dsl.Dataset],
    labels_path: dsl.Output[dsl.Dataset]
):

    import os
    import numpy as np
    import cv2
    from google.cloud import storage

    BUCKET_NAME = "mlops-imagenes-usuarios-vertex"
    clases = ["con-tapabocas", "sin-tapabocas"]

    data = []
    labels = []

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    for idx, clase in enumerate(clases):
        #blobs = bucket.list_blobs(prefix=clase)
        blobs = bucket.list_blobs(prefix=f"{clase}/")

        for blob in blobs:
            if blob.name.endswith(".jpg"):
                contenido = blob.download_as_bytes()

                np_arr = np.frombuffer(contenido, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                img = cv2.resize(img, (224, 224))
                img = img.astype("float32") / 255.0

                data.append(img)

                # one-hot encoding
                if idx == 0:
                    labels.append([1, 0])
                else:
                    labels.append([0, 1])

    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="float32")

    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)

  
    #np.save(data_path.path, data)
    #np.save(labels_path.path, labels)
    np.save(data_path.path + ".npy", data)
    np.save(labels_path.path + ".npy", labels)
    
@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "tensorflow",
        "numpy",
        "scikit-learn",
        "google-cloud-storage"
    ]
)
def entrenar_modelo(data_path: dsl.Input[dsl.Dataset],
                    labels_path: dsl.Input[dsl.Dataset]):

    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from google.cloud import storage
    import os

    
    data = np.load(data_path.path + ".npy")
    labels = np.load(labels_path.path + ".npy")

    print("Data :", data.shape)
    print("Labels :", labels.shape)

    trainX, testX, trainY, testY = train_test_split(
        data,
        labels,
        test_size=0.20,
        stratify=np.argmax(labels, axis=1),
        random_state=42
    )


    client = storage.Client()
    bucket = client.bucket("ml-models-mascarilla")

    local_model_path = "/tmp/model.keras"

    blob = bucket.blob("retrain/model_mask_covid.keras")
    blob.download_to_filename(local_model_path)

    print("Modelo .keras descargado")


    model = tf.keras.models.load_model(local_model_path)
    print("Modelo cargado correctamente")


    for layer in model.layers[-3:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        epochs=1,
        batch_size=32
    )

    print("Fine-tuning terminado")


    export_path = "/tmp/export_vertex"
    model.export(export_path)

    for root, dirs, files in os.walk(export_path):
        for file in files:
            local_file = os.path.join(root, file)
            remote_path = os.path.relpath(local_file, export_path)

            blob = bucket.blob(f"model_mask_covid_vertex/{remote_path}")
            blob.upload_from_filename(local_file)

    print("Modelo Vertex exportado")

    
    keras_path = "/tmp/model_retrain.keras"
    model.save(keras_path)

    blob = bucket.blob("model_mask_covid_retrain/model_mask_covid.keras")
    blob.upload_from_filename(keras_path)

    print("Modelo Keras guardado para reentrenamiento futuro")







@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "tensorflow",
        "numpy",
        "scikit-learn",
        "google-cloud-storage"
    ]
)
def evaluar_modelo(
    data_path: dsl.Input[dsl.Dataset],
    labels_path: dsl.Input[dsl.Dataset]
):
    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from google.cloud import storage
    import os


    data = np.load(data_path.path + ".npy")
    labels = np.load(labels_path.path + ".npy")

    _, testX, _, testY = train_test_split(
        data,
        labels,
        test_size=0.2,
        stratify=np.argmax(labels, axis=1),
        random_state=42
    )

    client = storage.Client()
    bucket = client.bucket("ml-models-mascarilla")

    local_model_path = "/tmp/model_vertex"
    os.makedirs(local_model_path, exist_ok=True)

    blobs = bucket.list_blobs(prefix="model_mask_covid_vertex/")

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        file_path = os.path.join(
            local_model_path,
            blob.name.replace("model_mask_covid_vertex/", "")
        )    
         
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob.download_to_filename(file_path)

        print("Modelo Vertex descargado correctamente")    

    model = tf.keras.layers.TFSMLayer(
    local_model_path,
    call_endpoint="serving_default"
    )
     
    preds = model(tf.constant(testX))
    preds = list(preds.values())[0].numpy()
    preds = np.argmax(preds, axis=1)
    
    true = np.argmax(testY, axis=1)
    
    report = classification_report(true, preds, output_dict=True)
    matrix = confusion_matrix(true, preds)
    
    print("Reporte:", report)
    print("Confusion Matrix:\n", matrix)

   
    precision_con = report["0"]["precision"]  # clase 0 = con mascarilla
    print("Precision (con mascarilla):", precision_con)

    if precision_con >= 0.80:
        print("ABRIR PUERTA")
        print("DEPLOY")
    else:
        print("NO ABRIR NO DEPLOY")




@dsl.pipeline(name="retrain-pipeline")
def retrain_pipeline():
    datos = cargar_datos()
    
    entrenamiento = entrenar_modelo(
        data_path=datos.outputs["data_path"],
        labels_path=datos.outputs["labels_path"]
    )
   
    evaluacion = evaluar_modelo(
        data_path=datos.outputs["data_path"],
        labels_path=datos.outputs["labels_path"]
    )