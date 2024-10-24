
import sqlite3
import json
from datetime import datetime

class AdvancedLogger:
    def __init__(self, db_path='training_log.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            iteration INTEGER,
            loss REAL,
            accuracy REAL,
            elo_rating INTEGER,
            additional_data TEXT
        )
        """)
        self.conn.commit()

    def log_training_progress(self, iteration, loss, accuracy, elo_rating, additional_data=None):
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO training_logs (iteration, loss, accuracy, elo_rating, additional_data)
        VALUES (?, ?, ?, ?, ?)
        """, (iteration, loss, accuracy, elo_rating, json.dumps(additional_data)))
        self.conn.commit()

    def get_training_progress(self, limit=100):
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT * FROM training_logs
        ORDER BY timestamp DESC
        LIMIT ?
        """, (limit,))
        return cursor.fetchall()

    def close(self):
        self.conn.close()

# Example usage
if __name__ == '__main__':
    logger = AdvancedLogger()
    logger.log_training_progress(1, 0.5, 0.7, 1200, {'batch_size': 64, 'learning_rate': 0.001})
    logger.log_training_progress(2, 0.4, 0.75, 1250, {'batch_size': 64, 'learning_rate': 0.001})
    print(logger.get_training_progress())
    logger.close()
