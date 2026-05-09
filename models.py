from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Float,BigInteger
from sqlalchemy.sql import func
from database import Base

class JobPost(Base):
    """
    Shadow model for the hr_app_job_post table.
    """
    __tablename__ = 'hr_app_job_post'

    # BigAutoField in Django maps to BigInteger in SQLAlchemy
    job_id = Column(BigInteger, primary_key=True, index=True)
    
    # CharField(max_length=350) maps to String(350)
    job_title = Column(String(350), nullable=False)
    
    # TextField maps to Text
    jobDescription = Column(Text, nullable=False)
    
    # ML Fields
    cluster = Column(Integer, nullable=True)
    clusterable_text = Column(Text, nullable=True)

class Employee(Base):
    """
    Shadow model for the Django 'Employee' model.
    """
    __tablename__ = 'hr_app_employee'

    employee_id_id = Column(Integer, primary_key=True, index=True)
        
    education = Column(Text, nullable=True)
    experience = Column(Text, nullable=True)
    awards = Column(Text, nullable=True)
    hobbies = Column(Text, nullable=True)
    skills = Column(Text, nullable=False)
    
    # ML specific fields
    cluster = Column(Integer, nullable=True)
    clusterable_text = Column(Text, nullable=True)

class Course(Base):
    """
    Shadow model for the hr_app_course table.
    """
    __tablename__ = 'hr_app_course'

    # BigAutoField in Django maps to BigInteger in SQLAlchemy
    course_id = Column(BigInteger, primary_key=True, index=True)
    
    # CharField(max_length=350) maps to String(350)
    courseTitle = Column(String(350), nullable=False)
    
    # TextField maps to Text
    description = Column(Text, nullable=False)
    
    # ML Fields
    cluster = Column(Integer, nullable=True)
    clusterable_text = Column(Text, nullable=True)

class ClusterRecord(Base):
    """
    Shadow model for the hr_app_cluster_records table.
    This acts as the 'History Log' of every training session.
    """
    __tablename__ = 'hr_app_cluster_records'

    id = Column(Integer, primary_key=True, index=True)
    
    # DateTime logic: Django's auto_now_add is handled by server_default
    added_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # ML Metrics
    silhouette_score = Column(Float)
    calinski_harabasz_score = Column(Float)
    inertia = Column(Float)
    
    # Parameters (matching the TextField types from your Django model)
    number_of_clusters = Column(Text)
    total_records = Column(Text)
    word2vec_vector_size = Column(Text)
    word2vec_window_size = Column(Text)
    word2vec_word_min_count_percentage = Column(Float)
    from_date = Column(Text)
    w2v_name = Column(String(500), nullable=True)
    kmeans_name = Column(String(500), nullable=True)
    experiment_id = Column(String(500), nullable=True)
    
    # The 'Applied' flag tells the UI which model is currently 'Live'
    applied = Column(Boolean, default=False)