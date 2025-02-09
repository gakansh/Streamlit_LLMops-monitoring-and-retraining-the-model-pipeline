import sqlite3
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
import hashlib
import os
import time  # Add time module for unique ID generation

# Configuration
DATABASE_NAME = "llm_monitoring.db"
MODEL_NAME = "facebook/bart-large-cnn"


# %% [markdown]
# ## 1. Database Setup
class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect(DATABASE_NAME, check_same_thread=False)  # FIX: Allow multi-threaded access
        self._create_tables()
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                input_text TEXT,
                summary TEXT,
                rating INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    def log_interaction(self, input_text, summary, rating=None):
        """Logs interactions (text, summary, rating) into the database."""
        unique_id = hashlib.md5(f"{input_text}{summary}{time.time()}".encode()).hexdigest()
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO interactions (id, input_text, summary, rating)
            VALUES (?, ?, ?, ?)
        ''', (unique_id, input_text, summary, rating))
        self.conn.commit()
        return unique_id
    
    def get_rated_samples(self):
        """Retrieve all rated samples from the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM interactions WHERE rating IS NOT NULL")
        return cursor.fetchall()  # Returns a list of rated samples
    

# %% [markdown]
# ## 2. Monitoring System
class MonitoringSystem:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'])
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.clusterer = KMeans(n_clusters=3)
        self.db = DatabaseManager()
    
    def log_interaction(self, input_text, summary, rating=None):
        # Generate a unique ID using input text, summary, and current timestamp
        unique_id = hashlib.md5(f"{input_text}{summary}{time.time()}".encode()).hexdigest()
        
        cursor = self.db.conn.cursor()
        cursor.execute('''
            INSERT INTO interactions (id, input_text, summary, rating)
            VALUES (?, ?, ?, ?)
        ''', (unique_id, input_text, summary, rating))
        self.db.conn.commit()
        return unique_id
    
    def calculate_metrics(self):
        df = pd.read_sql('SELECT * FROM interactions', self.db.conn)
        
        if df.empty:
            return {
                'rouge_score': 0,
                'drift_score': 0,
                'avg_rating': 0,
                'sample_count': 0,
                'rating_distribution': {}  # Ensure an empty dictionary is returned if no data
            }

        # ROUGE Scores
        df['rouge'] = df.apply(lambda x: self.scorer.score(x['input_text'], x['summary'])['rougeL'].fmeasure, axis=1)
        rouge_score = df['rouge'].mean()

        # Drift Detection
        if len(df) > 50:
            X = self.vectorizer.fit_transform(df['input_text'])
            self.clusterer.fit(X)
            drift_score = np.std(np.bincount(self.clusterer.labels_)) / len(df)
        else:
            drift_score = 0.0

        # Feedback Analysis
        avg_rating = df['rating'].mean() if not df['rating'].isnull().all() else 0
        sample_count = len(df)

        # Rating Distribution
        rating_distribution = df['rating'].value_counts().to_dict() if 'rating' in df.columns else {}

        return {
            'rouge_score': rouge_score,
            'drift_score': drift_score,
            'avg_rating': avg_rating,
            'sample_count': sample_count,
            'rating_distribution': rating_distribution  
        }
    

from datasets import Dataset

# %% [markdown]
# ## 3. Retraining Decision System


def tokenize(batch, tokenizer):
        return tokenizer(
        batch["input_text"],
        text_target=batch["summary"],
        max_length=1024,
        truncation=True,
        padding="max_length"
    )

class RetrainingDecider:
    def __init__(self):
        self.deployment_name = "gpt-4"  # Azure deployment name
    
    def should_retrain(self, metrics):
        prompt = f"""Analyze these model metrics:
        - Average Rating: {metrics['avg_rating']:.2f}/5
        - Sample Count: {metrics['sample_count']}
        - ROUGE-L: {metrics['rouge_score']:.2f}
        - Drift Score: {metrics['drift_score']:.2f}

        Should we retrain? Answer only YES or NO."""

       
        
        return "YES" in response.choices[0].message.content.upper()

# %% [markdown]
# ## 4. Retraining Pipeline
class RetrainingPipeline:
    def __init__(self):
        self.model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
        self.db = DatabaseManager()
    
    def prepare_data(self):
        df = pd.read_sql('''
            SELECT input_text, summary 
            FROM interactions 
            WHERE rating < 3 OR rating IS NULL
        ''', self.db.conn)
        return df
    
    # def tokenize(batch, tokenizer):
    #     return tokenizer(
    #     batch["input_text"],
    #     text_target=batch["summary"],
    #     max_length=1024,
    #     truncation=True,
    #     padding="max_length"
    # )
    
    def retrain(self):
        df = self.prepare_data()
        
        # Simple fine-tuning logic
        train_dataset = Dataset.from_pandas(df[['input_text', 'summary']])
        
        
        train_dataset = train_dataset.map(lambda batch: tokenize(batch, self.tokenizer), batched=True)

        
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            logging_steps=10
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        self.model.save_pretrained("./retrained_model")

# %% [markdown]
# ## 5. Complete Workflow
class LLMSystem:
    def __init__(self):
        self.monitor = MonitoringSystem()
        self.decider = RetrainingDecider()
        self.retrainer = RetrainingPipeline()
        self.tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
        self.model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    def summarize(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs["input_ids"], max_length=130)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def process_input(self, text, rating=None):
        # Generate summary
        summary = self.summarize(text)
        
        # Log interaction
        interaction_id = self.monitor.log_interaction(text, summary, rating)
        
        # Check metrics periodically
        if self.monitor.db.conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0] % 10 == 0:
            metrics = self.monitor.calculate_metrics()
            if self.decider.should_retrain(metrics):
                print("Initiating retraining...")
                self.retrainer.retrain()
        
        return summary

# %% [markdown]
# ## 6. Usage Example
if __name__ == "__main__":
    system = LLMSystem()
    
    # Example usage
    text = "Your long input text here..."
    summary = system.process_input(text)
    print(f"Generated Summary:\n{summary}")
    
    # Simulate user feedback
    system.process_input(text, rating=4)  # No summary needed for feedback
