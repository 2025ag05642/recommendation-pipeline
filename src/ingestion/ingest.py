# src/ingestion/ingest.py

import pandas as pd
import os 
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

def load_ratings():
    """Load ratings data from CSV"""
    logger.info("Loading ratings data...")
    df = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "ratings.csv")
    )
    logger.info(f"Loaded {len(df):,} ratings")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Shape: {df.shape}")
    return df

def load_movies():
    """Load movies data from CSV"""
    logger.info("Loading movies data...")
    df = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "movies.csv")
    )
    logger.info(f"Loaded {len(df):,} movies")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Shape: {df.shape}")
    return df


def load_tags():
    """Load tags data from CSV"""
    logger.info("Loading tags data...")
    df = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "tags.csv")
    )
    logger.info(f"Loaded {len(df):,} tags")
    logger.info(f"Shape: {df.shape}")
    return df

def load_links():
    """Load links data from CSV"""
    logger.info("Loading links data...")
    df = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "links.csv")
    )
    logger.info(f"Loaded {len(df):,} links")
    logger.info(f"Shape: {df.shape}")
    return df

def save_to_parquet(df, filename):
    """Save dataframe to parquet format"""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_PATH, filename)
    df.to_parquet(path, index=False)
    logger.info(f"Saved to {path}")

def ingest_all():
    """Main ingestion function"""
    logger.info("="*50)
    logger.info("Starting Data Ingestion...")
    logger.info("="*50)

    # Load all datasets
    ratings = load_ratings()
    movies = load_movies()
    tags = load_tags() 
    links = load_links()

    # Save as parquet
    save_to_parquet(ratings, "ratings.parquet")
    save_to_parquet(movies, "movies.parquet")
    save_to_parquet(tags, "tags.parquet")
    save_to_parquet(links, "links.parquet")

    logger.info("="*50)
    logger.info("Data Ingestion Complete!")
    logger.info(f"Ratings: {len(ratings):,} rows") 
    logger.info(f"Movies: {len(movies):,} rows")
    logger.info(f"Tags: {len(tags):,} rows") 
    logger.info(f"Links: {len(links):,} rows")
    logger.info("="*50)

    return ratings, movies, tags, links

if __name__ == "__main__":
    ingest_all()
