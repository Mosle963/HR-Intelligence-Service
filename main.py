from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import os
from fastapi import Security, HTTPException, status, Request, Depends
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded



# Local imports
from database import engine, SessionLocal,get_db
import models
from services.predictor import ml_state
from services.training_service import run_clustering_experiment,sync_all_records
from services.helpers import generate_experiment_id,get_storage_info, cleanup_old_models



# Create tables if they don't exist (Shadowing Django)
models.Base.metadata.create_all(bind=engine)



# 1. Define the Security Scheme
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 2. Authentication Logic
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == os.getenv("API_SECRET_KEY"):
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials"
    )

# 3. Middleware for IP Whitelisting (Optional but powerful)
class IPWhitelistMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        allowed_ips = os.getenv("ALLOWED_HOSTS", "127.0.0.1").split(",")
        client_ip = request.client.host
        if client_ip not in allowed_ips:
            return HTTPException(status_code=403, detail="IP not allowed")
        return await call_next(request)




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

app = FastAPI(title="HR Clustering API", lifespan=lifespan,dependencies=[Depends(get_api_key)])
app.add_middleware(IPWhitelistMiddleware)
limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.middleware("http")
async def limit_payload_size(request: Request, call_next):
    max_size = 1024 * 500  # 500KB limit
    size = request.headers.get("content-length")
    if size and int(size) > max_size:
        raise HTTPException(status_code=413, detail="Payload too large")
    
    return await call_next(request)
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
@limiter.limit("1/minute")
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
@limiter.limit("1/minute")
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
@limiter.limit("5/minute")
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
@limiter.limit("30/minute")
def get_training_status():
    from services.training_service import is_training_busy
    return {
        "is_busy": is_training_busy,
        "message": "System is training..." if is_training_busy else "Idle"
    }


@app.get("/syncing-status")
@limiter.limit("30/minute")
def get_syncing_status():
    from services.training_service import is_syncing_busy
    return {
        "is_busy": is_syncing_busy,
        "message": "Syncing is ongoing..." if is_syncing_busy else "Idle"
    }

@app.get("/storage-info")
def storage_info():
    """
    Returns the current disk space used by the ML models.
    """
    info = get_storage_info()
    return {
        "status": "success",
        "data": info
    }

@app.post("/cleanup-models")
def trigger_cleanup(keep_count: int = 5, db: Session = Depends(get_db)):
    """
    Frees up disk space by deleting unapplied, older model files.
    """
    try:
        result = cleanup_old_models(db, keep_count)
        return {
            "status": "success",
            "message": "Cleanup complete.",
            "details": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Cleanup failed: {str(e)}"
        }