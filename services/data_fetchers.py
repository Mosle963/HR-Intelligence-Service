from sqlalchemy.orm import Session

class PostgresTextStreamer:
    def __init__(self, db: Session, model_class, text_column_name: str, start_date=None):
        self.db = db
        self.model_class = model_class
        self.column_name = text_column_name
        self.start_date = start_date

    def __iter__(self):
        """
        This method is called every time you start a new loop 
        (e.g., build_vocab, train, and get_mean_vectors).
        """
        # 1. Start the Query
        query = self.db.query(getattr(self.model_class, self.column_name))
        
        # 2. Apply the Date Filter if provided
        # Assuming your Django models use 'added_date' or similar
        if self.start_date:
            query = query.filter(self.model_class.added_date >= self.start_date)
        
        # 3. Stream the results in chunks of 1000
        query = query.yield_per(1000)
        
        for row in query:
            text = getattr(row, self.column_name)
            if text:
                # Yielding the split list makes it 'Gensim-ready'
                yield text.split()