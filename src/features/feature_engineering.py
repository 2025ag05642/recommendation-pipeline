# src/features/feature_engineering.py

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

def load_data():  
    """Load all required data"""  
    train = pd.read_parquet(os.path.join(PROCESSED_PATH, "train_transformed.parquet"))  
    test = pd.read_parquet(os.path.join(PROCESSED_PATH, "test_transformed.parquet"))  
    movies = pd.read_parquet(os.path.join(FEATURES_PATH, "movies_encoded.parquet"))  
    return train, test, movies

def create_user_features(train):  
    """Create user features"""  
    logger.info("Creating user features...")

    max_timestamp = train['timestamp'].max()

    user_features = train.groupby('userId').agg(  
        user_avg_rating=('rating', 'mean'),  
        user_activity_count=('rating', 'count'),  
        user_min_rating=('rating', 'min'),  
        user_max_rating=('rating', 'max'),  
        user_std_rating=('rating', 'std'),  
        last_interaction=('timestamp', 'max'),  
        first_interaction=('timestamp', 'min')  
    ).reset_index()

    user_features['days_since_last_interaction'] = ((max_timestamp - user_features['last_interaction']) / 86400).round(0).astype(int)  
    user_features['user_avg_rating'] = user_features['user_avg_rating'].round(2)  
    user_features['user_std_rating'] = user_features['user_std_rating'].fillna(0).round(2)

    logger.info(f"✅ User features: {len(user_features):,} users")  
    logger.info(f"Columns: {user_features.columns.tolist()}")

    return user_features

def create_preferred_category(train, movies):  
    """Create preferred category feature for users"""  
    logger.info("Creating preferred category feature...")

    merged = train.merge(movies[['movieId', 'genres']], on='movieId', how='left')  
    merged['genre_list'] = merged['genres'].str.split('|')  
    merged_exploded = merged.explode('genre_list')

    user_genre_ratings = merged_exploded.groupby(['userId', 'genre_list'])['rating'].mean().reset_index()

    preferred_category = user_genre_ratings.loc[  
        user_genre_ratings.groupby('userId')['rating'].idxmax()  
    ][['userId', 'genre_list']].rename(columns={'genre_list': 'preferred_category'})

    logger.info(f"✅ Preferred category created for {len(preferred_category):,} users")

    return preferred_category

def create_item_features(train, movies):  
    """Create item features"""  
    logger.info("Creating item features...")

    item_features = train.groupby('movieId').agg(  
        item_avg_rating=('rating', 'mean'),  
        item_popularity=('rating', 'count'),  
        item_min_rating=('rating', 'min'),  
        item_max_rating=('rating', 'max'),  
        item_std_rating=('rating', 'std')  
    ).reset_index()

    item_features = item_features.merge(  
        movies[['movieId', 'title', 'genres', 'year']],  
        on='movieId',  
        how='left'  
    )

    item_features['item_avg_rating'] = item_features['item_avg_rating'].round(2)  
    item_features['item_std_rating'] = item_features['item_std_rating'].fillna(0).round(2)

    logger.info(f"✅ Item features: {len(item_features):,} items")  
    logger.info(f"Columns: {item_features.columns.tolist()}")

    return item_features

def create_feature_catalog(user_features, item_features, preferred_category):  
    """Create feature catalog documentation"""  
    logger.info("Creating feature catalog...")

    catalog = {  
        'Feature Name': [  
            'user_avg_rating',  
            'user_activity_count',  
            'user_std_rating',  
            'user_min_rating',  
            'user_max_rating',  
            'days_since_last_interaction',  
            'preferred_category',  
            'item_avg_rating',  
            'item_popularity',  
            'item_std_rating',  
            'item_min_rating',  
            'item_max_rating'  
        ],  
        'Type': [  
            'User', 'User', 'User', 'User', 'User',  
            'User', 'User',  
            'Item', 'Item', 'Item', 'Item', 'Item'  
        ],  
        'Description': [  
            'Average rating given by user',  
            'Total number of ratings by user',  
            'Standard deviation of user ratings',  
            'Minimum rating given by user',  
            'Maximum rating given by user',  
            'Days since user last interacted',  
            'Users most rated genre category',  
            'Average rating received by item',  
            'Total number of ratings for item',  
            'Standard deviation of item ratings',  
            'Minimum rating received by item',  
            'Maximum rating received by item'  
        ],  
        'Data Type': [  
            'float', 'int', 'float', 'float', 'float',  
            'int', 'string',  
            'float', 'int', 'float', 'float', 'float'  
        ],  
        'Used For': [  
            'Training & Prediction',  
            'Training & Prediction',  
            'Training & Prediction',  
            'Training & Prediction',  
            'Training & Prediction',  
            'Training & Prediction',  
            'Training & Prediction',  
            'Training & Prediction',  
            'Training & Prediction',  
            'Training & Prediction',  
            'Training & Prediction',  
            'Training & Prediction'  
        ]  
    }

    catalog_df = pd.DataFrame(catalog)  
    logger.info(f"✅ Feature catalog: {len(catalog_df)} features")  
    return catalog_df

def save_features(user_features, item_features, preferred_category, catalog_df):  
    """Save all features"""  
    os.makedirs(FEATURES_PATH, exist_ok=True)

    user_features.to_parquet(os.path.join(FEATURES_PATH, "user_features.parquet"), index=False)  
    item_features.to_parquet(os.path.join(FEATURES_PATH, "item_features.parquet"), index=False)  
    preferred_category.to_parquet(os.path.join(FEATURES_PATH, "preferred_category.parquet"), index=False)  
    catalog_df.to_csv(os.path.join(FEATURES_PATH, "feature_catalog.csv"), index=False)

    logger.info("✅ All features saved!")

def engineer_features():  
    """Main feature engineering function"""  
    logger.info("="*50)  
    logger.info("Starting Feature Engineering...")  
    logger.info("="*50)

    train, test, movies = load_data()

    user_features = create_user_features(train)  
    preferred_category = create_preferred_category(train, movies)  
    item_features = create_item_features(train, movies)

    user_features = user_features.merge(preferred_category, on='userId', how='left')

    catalog_df = create_feature_catalog(user_features, item_features, preferred_category)

    save_features(user_features, item_features, preferred_category, catalog_df)

    logger.info("="*50)  
    logger.info("Feature Engineering Complete!")  
    logger.info(f"User features: {user_features.shape}")  
    logger.info(f"Item features: {item_features.shape}")  
    logger.info(f"Feature catalog: {len(catalog_df)} features")  
    logger.info("="*50)  
    logger.info("\nFeature Catalog:")  
    logger.info("\n" + catalog_df.to_string())

    return user_features, item_features, catalog_df

if __name__ == "__main__":  
    engineer_features()  
