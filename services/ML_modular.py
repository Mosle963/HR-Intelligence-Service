from pathlib import Path
import gensim
from helpers import name_model
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
import joblib
import numpy as np

def get_mean_vectors(model, corpus):
    """
    Converts a stream of text into a single NumPy matrix of mean vectors.
    """
    vectors = []
    for words in corpus:
        # Get vectors for words that exist in the model's vocab
        valid_vectors = [model.wv[w] for w in words if w in model.wv]
        if valid_vectors:
            vectors.append(np.mean(valid_vectors, axis=0))
        else:
            # Handle empty/unknown sentences with a vector of zeros
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)


def train_word2vec(window, min_count, vector_size, corpus, workers=4):
    # 1. Initialize and Train
    model = gensim.models.Word2Vec(
        window=window,
        min_count=min_count,
        vector_size=vector_size,
        workers=workers
    )
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # 2. Path Handling
    folder = Path(__file__).resolve().parent / "models_storage" / "word2vec"
    folder.mkdir(parents=True, exist_ok=True) # Safety check
    
    # 3. Clean Naming
    name = name_model(model="w2v",w=window,mc=min_count,vc=vector_size,ext='model')
    save_path = folder / name
    
    # 4. Gensim-specific saving
    model.save(str(save_path))
    
    return {'save_path':str(save_path)}

def kmeans_fun(n_clusters, max_iter, n_init, vectors):
    model = KMeans(
        n_clusters=n_clusters, 
        init='k-means++', 
        max_iter=max_iter,
        n_init=n_init,
        random_state=42 # Added for reproducible results in testing
    )
    labels = model.fit_predict(vectors)

    # Path Handling
    folder = Path(__file__).resolve().parent / "models_storage" / "kmeans"
    folder.mkdir(parents=True, exist_ok=True)
    
    # Naming
    name = name_model(model="kmeans", ncluster=n_clusters, mxi=max_iter, ni=n_init, ext='joblib')
    save_path = folder / name
    
    # Save
    joblib.dump(model, save_path)  
 
    return {
        'labels': labels, 
        'inertia': float(model.inertia_), 
        'save_path': str(save_path),
    }

def sil_fun(vectors, labels):
    if len(set(labels)) < 2: return 0 
    score = silhouette_score(vectors, labels)
    return round(float(score), 3)

def ch_fun(vectors, labels):
    if len(set(labels)) < 2: return 0
    score = calinski_harabasz_score(vectors, labels)
    return round(float(score), 3)