# HR Intelligence Engine (FastAPI)

A high-performance machine learning microservice built to handle text preprocessing, vectorization, and unsupervised clustering for HR data (Job Posts, Resumes, and Courses). 

This service is designed to work as a standalone "Brain," decoupling heavy ML computations from the main web application (Django).

## 🛠 Key Features

* **RAM-Cached Inference:** Models are loaded into memory on startup for lightning-fast predictions (sub-100ms).
* **Background Training Orchestrator:** Heavy Word2Vec and KMeans training sessions run in the background to prevent API timeouts.
* **Hot-Swapping Webhooks:** Update the active model in real-time without restarting the server via a `/reload-models` endpoint.
* **Automatic Data Syncing:** When a new model is applied, the engine automatically re-clusters existing database records in a "slow and steady" background task.
* **PostgreSQL Integration:** Streams data directly from the database using SQLAlchemy for memory-efficient training.

## 🏗 Tech Stack

* **Framework:** FastAPI (Asynchronous Python)
* **ML Libraries:** Gensim (Word2Vec), Scikit-Learn (KMeans), Joblib
* **NLP:** spaCy (lemmatization and tech-token preservation)
* **Database:** SQLAlchemy (PostgreSQL)
* **Validation:** Pydantic

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone [https://github.com/your-username/HR_W2V_Kmeans_FastAPI.git](https://github.com/your-username/HR_W2V_Kmeans_FastAPI.git)
cd HR_W2V_Kmeans_FastAPI

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm