# src/transformation/transform.py

import pandas as pd  
import numpy as np  
import os  
import logging

logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)  
logger = logging.getLogger(__name__)

PROCESSED_PATH = "data/processed/"  
FEATURES_PATH = "data/features/"

def load_prepared_data():  
    """Load prepared data"""  
    train = pd.read_parquet(  
        os.path.join(PROCESSED_PATH, "train.parquet")  
    )  
    test = pd.read_parquet(  
        os.path.join(PROCESSED_PATH, "test.parquet")  
    )  
    movies = pd.read_parquet(  
        os.path.join(PROCESSED_PATH, "movies.parquet")  
    )  
    user_summary = pd.read_parquet(  
        os.path.join(PROCESSED_PATH, "user_summary.parquet")  
    )  
    item_popularity = pd.read_parquet(  
        os.path.join(PROCESSED_PATH, "item_popularity.parquet")  
    )  
    return train, test, movies, user_summary, item_popularity

def normalize_ratings(train, test):  
    """Normalize ratings to 0-1 scale"""  
    logger.info("Normalizing ratings...")

    min_rating = 0.5  
    max_rating = 5.0

    train = train.copy()  
    test = test.copy()

    train['rating_normalized'] = (  
        (train['rating'] - min_rating) /   
        (max_rating - min_rating)  
    ).round(4)

    test['rating_normalized'] = (  
        (test['rating'] - min_rating) /   
        (max_rating - min_rating)  
    ).round(4)

    logger.info("✅ Ratings normalized to 0-1 scale")  
    return train, test

def encode_genres(movies):  
    """One-hot encode movie genres"""  
    logger.info("Encoding genres...")

    # Split genres  
    genres_split = movies['genres'].str.get_dummies('|')

    # Combine with movies  
    movies_encoded = pd.concat(  
        [movies, genres_split], axis=1  
    )

    logger.info(  
        f"✅ Genres encoded: {genres_split.shape[1]} genres"  
    )  
    logger.info(f"Genre columns: {genres_split.columns.tolist()}")

    return movies_encoded

def extract_year_from_title(movies):  
    """Extract year from movie title"""  
    logger.info("Extracting year from titles...")

    movies = movies.copy()  
    movies['year'] = movies['title'].str.extract(  
        r'\((\d{4})\)'  
    )  
    movies['year'] = pd.to_numeric(  
        movies['year'], errors='coerce'  
    )  
    movies['year'] = movies['year'].fillna(0).astype(int)

    logger.info(  
        f"✅ Year extracted for "  
        f"{(movies['year'] > 0).sum():,} movies"  
    )  
    return movies

def create_rating_categories(df):  
    """Create rating categories"""  
    logger.info("Creating rating categories...")

    df = df.copy()  
    df['rating_category'] = pd.cut(  
        df['rating'],  
        bins=[0, 2, 3, 4, 5],  
        labels=['poor', 'average', 'good', 'excellent']  
    )

    logger.info("✅ Rating categories created")  
    logger.info(  
        df['rating_category'].value_counts().to_string()  
    )  
    return df

def transform_all():  
    """Main transformation function"""  
    logger.info("="*50)  
    logger.info("Starting Data Transformation...")  
    logger.info("="*50)

    # Load data
    data = load_prepared_data()  
    train, test, movies, user_summary, item_popularity = data  


    # Transformations  
    train, test = normalize_ratings(train, test)  
    movies_encoded = encode_genres(movies)  
    movies_encoded = extract_year_from_title(movies_encoded)  
    train = create_rating_categories(train)  
    test = create_rating_categories(test)

    # Save transformed data  
    os.makedirs(FEATURES_PATH, exist_ok=True)

    train.to_parquet(  
        os.path.join(PROCESSED_PATH, "train_transformed.parquet"),  
        index=False  
    )  
    test.to_parquet(  
        os.path.join(PROCESSED_PATH, "test_transformed.parquet"),  
        index=False  
    )  
    movies_encoded.to_parquet(  
        os.path.join(FEATURES_PATH, "movies_encoded.parquet"),  
        index=False  
    )

    logger.info("="*50)  
    logger.info("Data Transformation Complete!")  
    logger.info(f"Train transformed: {len(train):,} rows")  
    logger.info(f"Test transformed: {len(test):,} rows")  
    logger.info(  
        f"Movies encoded: {movies_encoded.shape[1]} columns"  
    )  
    logger.info("="*50)

    return train, test, movies_encoded

if __name__ == "__main__":  
    transform_all()  
