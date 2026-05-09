from database import SessionLocal
from services import ml_modular, data_fetchers
import models
import datetime

is_training_busy = False

def run_clustering_experiment(
    start_date, 
    min_count_pct, 
    w2v_window, 
    w2v_size, 
    n_clusters_list, # List of ints: [3, 5, 8, 10]
    experiment_id,
    max_iter=300,
    
):
    
    global is_training_busy
    if is_training_busy:
        return # Safety skip
    
    is_training_busy = True
    db = SessionLocal()
    try:
        # 1. Setup Streamer (Filtered by Date)
        # We assume your streamer can handle filters (we can add a .filter() to the query)
        streamer = data_fetchers.PostgresTextStreamer(
            db=db, 
            model_class=models.JobPost, 
            text_column_name="clusterable_text",
            from_date=start_date 
        )

        # 2. Pre-calculate Count (Need total length for the percentage)
        total_records = db.query(models.JobPost).filter(models.JobPost.added_date >= start_date).count()
        word2vec_min_count = max(1, int(total_records * min_count_pct))

        # 3. Train Word2Vec (Streaming)
        w2v_results = ml_modular.train_word2vec(
            window=w2v_window,
            min_count=word2vec_min_count,
            vector_size=w2v_size,
            corpus=streamer
        )
        
        # Load the newly trained model to create vectors
        from gensim.models import Word2Vec
        w2v_model = Word2Vec.load(w2v_results['save_path'])

        # 4. Create the NumPy Matrix for KMeans (The 'Bucket')
        vectors = ml_modular.get_mean_vectors(w2v_model, streamer)

        # 5. Loop through different cluster counts (The Elbow Method)
        for n in n_clusters_list:
            # Train KMeans
            km_results = ml_modular.kmeans_fun(
                n_clusters=n, 
                max_iter=max_iter, 
                n_init=10, 
                vectors=vectors
            )

            # Calculate Scores
            s_score = ml_modular.sil_fun(vectors, km_results['labels'])
            ch_score = ml_modular.ch_fun(vectors, km_results['labels'])

            # 6. Save to Cluster Records
            new_record = models.ClusterRecord(
                added_date=datetime.now(),
                silhouette_score=s_score,
                calinski_harabasz_score=ch_score,
                inertia=km_results['inertia'],
                number_of_clusters=str(n),
                total_records=str(total_records),
                word2vec_vector_size=str(w2v_size),
                word2vec_window_size=str(w2v_window),
                word2vec_word_min_count_percentage=min_count_pct,
                from_date=str(start_date),
                applied=False,
                w2v_name=w2v_results['model_name'], # Need to return this from W2V fun
                kmeans_name=km_results['model_name'],
                experiment_id = experiment_id
            )
            db.add(new_record)
        
        db.commit()
        return {"status": "success", "experiments_run": len(n_clusters_list)}

    except Exception as e:
        db.rollback()
        raise e
    finally:
        is_training_busy = False
        db.close()

is_syncing_busy = False # Separate lock so we don't mix training and syncing
def sync_all_records():
    global is_syncing_busy
    is_syncing_busy = True
    db = SessionLocal()
    try:
        from .predictor import ml_state
        from models import Employee, Job_Post, Course
        
        # We process in chunks so we don't overload the RAM
        CHUNK_SIZE = 500
        
        for model_class in [Employee, Job_Post, Course]:
            offset = 0
            while True:
                # 1. Fetch a chunk of records
                records = db.query(model_class).offset(offset).limit(CHUNK_SIZE).all()
                if not records:
                    break
                
                # 2. Predict and Update
                for record in records:
                    if record.clusterable_text:
                        # Use the RAM model!
                        new_cluster = ml_state.predict(record.clusterable_text, is_already_clean=True)
                        record.cluster = new_cluster
                
                # 3. Save this chunk to the DB
                db.commit()
                offset += CHUNK_SIZE
                print(f"Synced {offset} records for {model_class.__name__}...")

    except Exception as e:
        print(f"Sync failed: {e}")
        db.rollback()
    finally:
        is_syncing_busy = False