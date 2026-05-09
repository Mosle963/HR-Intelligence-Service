from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

# Local imports
from database import engine, SessionLocal
import models
from services.predictor import ml_state
from services.training_service import run_clustering_experiment,sync_all_records
from services.helpers import generate_experiment_id

# Create tables if they don't exist (Shadowing Django)
models.Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------
# 1. Lifespan (Startup/Shutdown Events)
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the applied models into RAM
    db = SessionLocal()
    try:
        print("Starting up The Brain...")
        ml_state.load_applied_models(db)
    finally:
        db.close()
    yield
    # Shutdown: Clean up resources if needed
    print("Shutting down The Brain...")

app = FastAPI(title="HR Clustering API", lifespan=lifespan)

# ---------------------------------------------------------
# 2. Database Dependency
# ---------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------
# 3. Pydantic Schemas (Input Validation)
# ---------------------------------------------------------
class PredictRequest(BaseModel):
    text: str
    is_already_clean: bool = False

class TrainRequest(BaseModel):
    start_date: datetime
    min_count_pct: float
    w2v_window: int
    w2v_size: int
    n_clusters_list: List[int]


class ProcessResponse(BaseModel):
    clusterable_text: str
    cluster_id: int

# ---------------------------------------------------------
# 4. Endpoints
# ---------------------------------------------------------

@app.post("/predict")
def predict_cluster(request: PredictRequest):
    """
    Lightning-fast prediction using pure RAM. 
    No database queries.
    """
    if not ml_state.is_ready:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")
    
    cluster_id = ml_state.predict(request.text, request.is_already_clean)
    
    if cluster_id == -1:
        raise HTTPException(status_code=400, detail="Could not process text.")
        
    return {"cluster_id": cluster_id}


@app.post("/train")
def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Starts the heavy ML training in the background.
    Returns an Experiment ID immediately so Django doesn't timeout.
    """
    # Generate the ID here so we can return it to Django immediately
    experiment_id = generate_experiment_id()
    
    background_tasks.add_task(
        run_clustering_experiment,
        experiment_id=experiment_id, 
        start_date=request.start_date,
        min_count_pct=request.min_count_pct,
        w2v_window=request.w2v_window,
        w2v_size=request.w2v_size,
        n_clusters_list=request.n_clusters_list
    )
    
    return {
        "status": "Processing Started",
        "experiment_id": experiment_id,
        "message": "Check the database later for results."
    }


@app.post("/reload-models")
def reload_applied_models(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    1. Loads new models into RAM.
    2. Starts a background task to re-cluster every record in the DB.
    """
    success = ml_state.load_applied_models(db)
    
    if success:
        # Start the slow and steady sync
        background_tasks.add_task(sync_all_records, db)      
        return {
            "status": "success", 
            "message": "Models hot-swapped. Background sync started."
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to load models.")
    

@app.post("/process-and-predict", response_model=ProcessResponse)
def process_and_predict(request: PredictRequest):
    """
    Combined endpoint to minimize network trips.
    Cleans raw text and predicts the cluster in one go.
    """
    from services.preprocessor import preprocess
    
    # 1. Preprocess
    clean_text = preprocess(request.text)
    
    # 2. Predict
    if not ml_state.is_ready:
        # Fallback if models aren't loaded
        return ProcessResponse(clusterable_text=clean_text, cluster_id=-1)
    
    cluster_id = ml_state.predict(clean_text, is_already_clean=True)
    
    return ProcessResponse(clusterable_text=clean_text, cluster_id=cluster_id)

@app.get("/training-status")
def get_training_status():
    from services.training_service import is_training_busy
    return {
        "is_busy": is_training_busy,
        "message": "System is training..." if is_training_busy else "Idle"
    }


@app.get("/syncing-status")
def get_syncing_status():
    from services.training_service import is_syncing_busy
    return {
        "is_busy": is_syncing_busy,
        "message": "Syncing is ongoing..." if is_syncing_busy else "Idle"
    }