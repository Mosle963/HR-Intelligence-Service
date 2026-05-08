from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
import models
from database import get_db

app = FastAPI(title="The Brain - ML Microservice")

@app.get("/")
def read_root():
    return {"status": "online", "message": "The Brain is ready for clustering"}

@app.get("/test-orm")
def test_orm(db: Session = Depends(get_db)):
    """
    Verifies that FastAPI can actually read the data Django wrote.
    """
    try:
        # We use .job_id and .job_title because that's what we defined in models.py
        count = db.query(models.JobPost).count()
        first_job = db.query(models.JobPost).first()
        
        return {
            "database_connected": True,
            "total_jobs_in_db": count,
            "sample_data": {
                "id": first_job.job_id if first_job else None,
                "title": first_job.job_title if first_job else "No jobs found"
            }
        }
    except Exception as e:
        return {
            "database_connected": False,
            "error": str(e)
        }

@app.get("/test-employees")
def test_employees(db: Session = Depends(get_db)):
    """
    Verify we can see the employee table and the user IDs.
    """
    count = db.query(models.Employee).count()
    first_emp = db.query(models.Employee).first()
    
    return {
        "total_employees": count,
        "first_employee_id": first_emp.employee_id_id if first_emp else None
    }