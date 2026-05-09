import datetime
import uuid
from datetime import datetime

import os
from pathlib import Path

def cleanup_old_models(keep_count=5):
    """Deletes old models that aren't applied and exceed the keep_count."""
    # We'll need to query the DB to see which filenames are 'Applied'
    # and then delete the ones on disk that aren't in that list 
    # and are older than the last N records.
    pass # We can implement the logic once the DB is stable

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