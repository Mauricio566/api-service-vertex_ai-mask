from kfp import compiler
from pipeline import retrain_pipeline

compiler.Compiler().compile(
    pipeline_func=retrain_pipeline,
    package_path="retrain_pipeline.json"
)

print("✔ Pipeline compilado: retrain_pipeline.json")