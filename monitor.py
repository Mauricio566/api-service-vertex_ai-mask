from google.cloud import bigquery
from google.cloud import aiplatform


client = bigquery.Client()

PROJECT_ID = "project-a5d39078-68b8-4a9c-a89"
REGION = "us-central1"   
PIPELINE_ROOT = "gs://mlops-pipeline/pipeline_root"  

def ejecutar_pipeline():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.PipelineJob(
        display_name="retrain-job",
        template_path="retrain_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
    )

    job.run(sync=False) 
    print("🚀 Pipeline lanzado en Vertex AI")


query = """
SELECT
  COUNT(*) as total,
  COUNTIF(confidence < 0.90) as low_conf,
  COUNTIF(confidence < 0.90) / COUNT(*) as ratio,
  CASE 
    WHEN COUNTIF(confidence < 0.90) / COUNT(*) > 0.20 THEN "DEGRADADO"
    ELSE "OK"
  END as estado
FROM `project-a5d39078-68b8-4a9c-a89.ml_monitoring.predictions`
"""

# 3. Ejecutar consulta
query_job = client.query(query)
results = query_job.result()

# 4. Leer resultado
for row in results:
    total = row["total"]
    low_conf = row["low_conf"]
    ratio = row["ratio"]
    estado = row["estado"]

 
    if estado == "DEGRADADO":
        print(" MODELO DEGRADADO")
        ejecutar_pipeline()
    else:
        print("Modelo OK")
        #ejecutar_pipeline()
