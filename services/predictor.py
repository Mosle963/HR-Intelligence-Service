import joblib
from gensim.models import Word2Vec
from pathlib import Path
from sqlalchemy.orm import Session
import models
from .preprocessor import preprocess 

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent # Pointing to the root FastAPI folder
MODELS_DIR = BASE_DIR / 'services' / 'models_storage'

class ModelManager:
    def __init__(self):
        # These live in your server's RAM
        self.kmeans_model = None
        self.word2vec_model = None
        self.is_ready = False

    def load_applied_models(self, db: Session):
        """Called at startup, and whenever an Admin clicks 'Apply'"""
        # 1. Ask the DB which experiment is currently "Applied"
        applied_record = db.query(models.ClusterRecord).filter(models.ClusterRecord.applied == True).first()
        
        if not applied_record:
            print("No applied model in DB. Falling back to BASE model...")
            w2v_path = MODELS_DIR / 'base' / 'base_w2v.model'
            kmeans_path = MODELS_DIR / 'base' / 'base_kmeans.joblib'
        else:
            w2v_path = MODELS_DIR / 'word2vec' / applied_record.w2v_name
            kmeans_path = MODELS_DIR / 'kmeans' / applied_record.kmeans_name

        try:
            
            # 3. Load into RAM (replacing the old ones automatically)
            self.word2vec_model = Word2Vec.load(str(w2v_path))
            self.kmeans_model = joblib.load(kmeans_path)
            
            self.is_ready = True
            print(f"Successfully loaded models from Experiment: {applied_record.experiment_id}")
            return True
            
        except Exception as e:
            print(f"Critical Error loading models: {e}")
            self.is_ready = False
            return False

    def predict(self, text: str, is_already_clean: bool = False) -> int:
        """The lightning-fast prediction function"""
        if not self.is_ready:
            return -1 # System not ready
        
        try:
            if is_already_clean:
                tokens = text.split()
            else:
                cleaned_text = preprocess(text)
                tokens = cleaned_text.split()
            
            if not tokens:
                return -1

            # Gensim's get_mean_vector is perfect here!
            w2v_vector = self.word2vec_model.wv.get_mean_vector(tokens)
            
            # Reshape for sklearn
            vector_2d = w2v_vector.reshape(1, -1)
            label = self.kmeans_model.predict(vector_2d)
            
            return int(label[0])
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return -1

# ---------------------------------------------------------
# Create a SINGLE instance of this manager for the whole app
# ---------------------------------------------------------
ml_state = ModelManager()