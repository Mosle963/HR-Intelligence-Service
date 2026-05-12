import datetime
import uuid
from datetime import datetime
import os
from pathlib import Path
from sqlalchemy.orm import Session
import models


def name_model(**kwargs):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = []
    # Use .items() to iterate through keys and values
    for k, v in kwargs.items():
        if k == 'ext':
            continue
        parts.append(f"{k}_{v}")
    
    name = "_".join(parts) + f"_{timestamp}.{kwargs.get('ext', 'joblib')}"
    return name

def generate_experiment_id():
    """
    Generates a unique, URL-friendly experiment ID.
    Example output: EXP-20260508-A4B2C
    """
    # 1. Get the current date (YearMonthDay)
    date_str = datetime.now().strftime("%Y%m%d")
    
    # 2. Get a short unique hash (first 5 characters of a random UUID)
    unique_suffix = uuid.uuid4().hex[:5].upper()
    
    return f"EXP-{date_str}-{unique_suffix}"


MODELS_DIR = Path("services/models_storage")
W2V_DIR = MODELS_DIR / "word2vec"
KMEANS_DIR = MODELS_DIR / "kmeans"

def get_dir_size(directory: Path) -> float:
    """Returns the size of a directory in Megabytes (MB)."""
    if not directory.exists():
        return 0.0
    total_size = sum(f.stat().st_size for f in directory.glob('**/*') if f.is_file())
    return round(total_size / (1024 * 1024), 2)

def get_storage_info() -> dict:
    """Gathers storage metrics for the admin dashboard."""
    return {
        "word2vec_mb": get_dir_size(W2V_DIR),
        "kmeans_mb": get_dir_size(KMEANS_DIR),
        "total_mb": get_dir_size(MODELS_DIR)
    }

def cleanup_old_models(db: Session, keep_count: int = 5) -> dict:
    """
    Deletes models that are NOT applied and NOT in the most recent 'keep_count'.
    Safely handles Gensim's associated .npy files.
    """
    # 1. Identify which records to KEEP
    # Get the currently applied record
    applied_record = db.query(models.ClusterRecord).filter(models.ClusterRecord.applied == True).first()
    
    # Get the most recent N records (regardless of applied status)
    recent_records = db.query(models.ClusterRecord).order_by(models.ClusterRecord.created_at.desc()).limit(keep_count).all()
    
    # Build a set of filenames we MUST NOT delete
    keep_w2v_names = {record.w2v_name for record in recent_records if record.w2v_name}
    keep_kmeans_names = {record.kmeans_name for record in recent_records if record.kmeans_name}
    
    if applied_record:
        if applied_record.w2v_name: keep_w2v_names.add(applied_record.w2v_name)
        if applied_record.kmeans_name: keep_kmeans_names.add(applied_record.kmeans_name)

    files_deleted = 0
    bytes_freed = 0

    # 2. Purge KMeans (Simple 1-to-1 files)
    if KMEANS_DIR.exists():
        for file_path in KMEANS_DIR.glob('*'):
            if file_path.is_file() and file_path.name not in keep_kmeans_names:
                bytes_freed += file_path.stat().st_size
                file_path.unlink()
                files_deleted += 1

    # 3. Purge Word2Vec (Multi-file complexity)
    if W2V_DIR.exists():
        for file_path in W2V_DIR.glob('*'):
            if not file_path.is_file():
                continue
            
            # Gensim files look like "exp_123.model", "exp_123.model.npy"
            # We check if this file starts with any of the names we want to KEEP
            is_kept = any(file_path.name.startswith(keep_name) for keep_name in keep_w2v_names)
            
            if not is_kept:
                bytes_freed += file_path.stat().st_size
                file_path.unlink()
                files_deleted += 1

    return {
        "files_deleted": files_deleted,
        "space_freed_mb": round(bytes_freed / (1024 * 1024), 2)
    }