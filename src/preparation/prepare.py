# src/preparation/prepare.py

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

def load_validated_data():  
    """Load validated parquet files"""  
    ratings = pd.read_parquet(  
        os.path.join(PROCESSED_PATH, "ratings.parquet")  
    )  
    movies = pd.read_parquet(  
        os.path.join(PROCESSED_PATH, "movies.parquet")  
    )  
    return ratings, movies

def clean_data(ratings, movies):  
    """Clean and standardize data"""  
    logger.info("Cleaning data...")

    # Remove duplicates  
    ratings = ratings.drop_duplicates()  
    movies = movies.drop_duplicates()

    # Handle missing values  
    ratings = ratings.dropna()  
    movies = movies.dropna()

    # Convert timestamp to datetime  
    ratings['datetime'] = pd.to_datetime(  
        ratings['timestamp'], unit='s'  
    )

    # Standardize genres  
    movies['genres'] = movies['genres'].str.strip()  
    movies['genres'] = movies['genres'].replace(  
        '(no genres listed)', 'Unknown'  
    )

    logger.info(f"Clean ratings: {len(ratings):,}")  
    logger.info(f"Clean movies: {len(movies):,}")

    return ratings, movies

def time_based_split(ratings, test_ratio=0.2):  
    """Split data based on time"""  
    logger.info("Creating time-based train/test split...")

    # Sort by timestamp  
    ratings_sorted = ratings.sort_values('timestamp')

    # Split point  
    split_idx = int(len(ratings_sorted) * (1 - test_ratio))  
    split_timestamp = ratings_sorted.iloc[split_idx]['timestamp']

    # Split  
    train = ratings_sorted[  
        ratings_sorted['timestamp'] < split_timestamp  
    ]  
    test = ratings_sorted[  
        ratings_sorted['timestamp'] >= split_timestamp  
    ]

    logger.info(f"Train set: {len(train):,} ratings")  
    logger.info(f"Test set: {len(test):,} ratings")  
    logger.info(  
        f"Split timestamp: "  
        f"{pd.to_datetime(split_timestamp, unit='s')}"  
    )

    return train, test

def create_user_activity_summary(ratings):  
    """Create user activity summary"""  
    logger.info("Creating user activity summary...")

    user_summary = ratings.groupby('userId').agg(  
        total_ratings=('rating', 'count'),  
        avg_rating=('rating', 'mean'),  
        min_rating=('rating', 'min'),  
        max_rating=('rating', 'max'),  
        first_rating=('timestamp', 'min'),  
        last_rating=('timestamp', 'max')  
    ).reset_index()

    user_summary['avg_rating'] = user_summary[  
        'avg_rating'  
    ].round(2)

    logger.info(  
        f"User activity summary: {len(user_summary):,} users"  
    )  
    return user_summary

def create_item_popularity(ratings, movies):  
    """Create item popularity dataset"""  
    logger.info("Creating item popularity...")

    item_popularity = ratings.groupby('movieId').agg(  
        rating_count=('rating', 'count'),  
        avg_rating=('rating', 'mean'),  
        total_rating=('rating', 'sum')  
    ).reset_index()

    # Merge with movies  
    item_popularity = item_popularity.merge(  
        movies[['movieId', 'title', 'genres']],  
        on='movieId',  
        how='left'  
    )

    # Popularity score  
    item_popularity['popularity_score'] = (  
        item_popularity['rating_count'] *   
        item_popularity['avg_rating']  
    ).round(2)

    item_popularity = item_popularity.sort_values(  
        'popularity_score', ascending=False  
    )

    logger.info(  
        f"Item popularity: {len(item_popularity):,} items"  
    )  
    return item_popularity

def create_user_item_matrix(ratings):  
    """Create user-item interaction matrix"""  
    logger.info("Creating user-item interaction matrix...")

    matrix = ratings.pivot_table(  
        index='userId',  
        columns='movieId',  
        values='rating',  
        fill_value=0  
    )

    logger.info(f"Matrix shape: {matrix.shape}")  
    sparsity = (  
        1 - ratings.shape[0] /   
        (matrix.shape[0] * matrix.shape[1])  
    ) * 100  
    logger.info(f"Sparsity: {sparsity:.2f}%")

    return matrix

def prepare_all():  
    """Main preparation function"""  
    logger.info("="*50)  
    logger.info("Starting Data Preparation...")  
    logger.info("="*50)

    # Load data  
    ratings, movies = load_validated_data()

    # Clean data  
    ratings, movies = clean_data(ratings, movies)

    # Time-based split  
    train, test = time_based_split(ratings)

    # Create derived datasets  
    user_summary = create_user_activity_summary(ratings)  
    item_popularity = create_item_popularity(ratings, movies)  
    user_item_matrix = create_user_item_matrix(train)

    # Save all datasets  
    train.to_parquet(  
        os.path.join(PROCESSED_PATH, "train.parquet"),  
        index=False  
    )  
    test.to_parquet(  
        os.path.join(PROCESSED_PATH, "test.parquet"),  
        index=False  
    )  
    user_summary.to_parquet(  
        os.path.join(PROCESSED_PATH, "user_summary.parquet"),  
        index=False  
    )  
    item_popularity.to_parquet(  
        os.path.join(PROCESSED_PATH, "item_popularity.parquet"),  
        index=False  
    )  
    user_item_matrix.to_parquet(  
        os.path.join(PROCESSED_PATH, "user_item_matrix.parquet")  
    )

    logger.info("="*50)  
    logger.info("Data Preparation Complete!")  
    logger.info(f"Train: {len(train):,} ratings")  
    logger.info(f"Test: {len(test):,} ratings")  
    logger.info(  
        f"Users: {user_summary.shape[0]:,}"  
    )  
    logger.info(  
        f"Items: {item_popularity.shape[0]:,}"  
    )  
    logger.info("="*50)

    return train, test, user_summary, item_popularity

if __name__ == "__main__":  
    prepare_all()  
