import mlflow
import os
from mlflow.tracking import MlflowClient

# --- Configuration ---
# In a pipeline, we pass this via environment variable.
# For manual testing, we try to detect the latest run.
MODEL_NAME = "ChurnModel"
EXPERIMENT_NAME = "Churn_Prediction_Basic"


def print_model_version_info(mv):
    print(f"Name: {mv.name} | Version: {mv.version} | Status: {mv.status} | Run ID: {mv.run_id} | Source: {mv.source} | Creation Time: {mv.creation_timestamp} | Last Updated: {mv.last_updated_timestamp}")
    print(f"Description: {mv.description}")

def get_latest_run_id():
    """Detect the last successful run in the experiment."""
    try:
        last_run = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME], 
            filter_string="tags.mlflow.runName = 'Model_Training'",
            order_by=["start_time DESC"], 
            max_results=1
        )
        if not last_run.empty:
            return last_run.iloc[0].run_id
    except Exception:
        pass
    return None

def register(run_id=None):
    # Evaluate at runtime to pick up values set by pipeline.py
    if not run_id:
        print("MLFLOW_RUN_ID not set. Attempting to auto-detect latest run...")
        run_id = get_latest_run_id()
        print(f"found run_id: {run_id}")

    if not run_id:
        run_id = os.getenv("MLFLOW_RUN_ID") 
        pass
            
    if not run_id:
        print("Error: Could not find any runs to register. Please run train.py first.")
        return

    print(f"Registering model from Run ID: {run_id} as '{MODEL_NAME}'...")
    
    # 1. Register Model
    # Inside Docker (mlflow run), the volume is mounted at /mlflow/tmp/mlruns
    # But the metadata stores the host path. We need to help MLflow find it.
    if os.path.exists("/mlflow/tmp/mlruns"):
        model_uri = f"file:///mlflow/tmp/mlruns/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}/{run_id}/artifacts/model"

        # On est dans le container → réécrire avec le vrai chemin hôte
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        host_mlruns = os.environ.get("HOST_MLRUNS_PATH", "/home/ubuntu/MLflow_churn_prediction/mlruns") #"/mlflow/tmp/mlruns")
        #host_mlruns = os.environ.get("HOST_MLRUNS_PATH", "/mlflow/tmp/mlruns")
        model_uri = f"file://{host_mlruns}/{experiment.experiment_id}/{run_id}/artifacts/model"        
    else:
        model_uri = f"runs:/{run_id}/model"

    # Insert your code here  Instanciez le client 
    client = MlflowClient()

    # Lister toutes les versions enregistrées
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        print(f"Version: {mv.version}, Stage: {mv.current_stage}, Status: {mv.status}, Run ID: {mv.run_id}")

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Registry URI:", mlflow.get_registry_uri())


    # Check if model already exists, if not create it
    try:
        # Insert your code here Initialiser le client et préparer le registre
        client.get_registered_model(MODEL_NAME) 
    except Exception:
        print(f"Creating registered model '{MODEL_NAME}'...")
        # Insert your code here 
        client.create_registered_model(MODEL_NAME)
    
    # Insert your code here Création de la version   
    model_details = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_id
    )

    print(f"Model registered pending. Version: {model_details.version}")
    # 2. Transition to Staging
    # Insert your code here - Attendre que la version soit en état READY avant de transitionner
    max_wait = 60  # secondes
    elapsed = 0
    while model_details.status != "READY" and elapsed < max_wait:
        print(f"wait time for model version to be ready")
        time.sleep(3)
        elapsed +=3
        model_details = client.get_model_version(MODEL_NAME, model_details.version)
    print(f"Model registered completed. Version: {model_details.version}")
    
    print(f"Transitioning version {model_details.version} to Staging...")
    # Insert your code here
    mv=client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_details.version,
        stage="Staging"
    )

    print_model_version_info(mv)

if __name__ == "__main__":
    register()
