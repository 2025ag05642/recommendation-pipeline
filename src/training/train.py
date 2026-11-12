# src/training/train.py

import pandas as pd  
import numpy as np  
import os  
import logging  
import mlflow  
import mlflow.sklearn  
from surprise import SVD, Dataset, Reader  
from surprise.model_selection import cross_validate  
from collections import defaultdict

logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)  
logger = logging.getLogger(__name__)

PROCESSED_PATH = "data/processed/"  
FEATURES_PATH = "data/features/"  
MODELS_PATH = "models/"

def load_data():  
    """Load train and test data"""  
    train = pd.read_parquet(os.path.join(PROCESSED_PATH, "train_transformed.parquet"))  
    test = pd.read_parquet(os.path.join(PROCESSED_PATH, "test_transformed.parquet"))  
    item_popularity = pd.read_parquet(os.path.join(PROCESSED_PATH, "item_popularity.parquet"))  
    movies = pd.read_parquet(os.path.join(FEATURES_PATH, "movies_encoded.parquet"))  
    return train, test, item_popularity, movies

def train_popularity_model(train, item_popularity):  
    """Train popularity-based recommender"""  
    logger.info("Training Model 1: Popularity-based Recommender...")

    with mlflow.start_run(run_name="popularity_model"):  
        mlflow.log_param("model_type", "popularity_based")  
        mlflow.log_param("metric", "rating_count")  
        mlflow.log_param("train_size", len(train))

        top_movies = item_popularity.nlargest(100, 'rating_count')

        mlflow.log_metric("num_recommendations", len(top_movies))  
        mlflow.log_metric("avg_rating", top_movies['avg_rating'].mean())  
        mlflow.log_metric("total_movies", len(item_popularity))

        top_movies.to_csv(os.path.join(MODELS_PATH, "popularity_model.csv"), index=False)  
        mlflow.log_artifact(os.path.join(MODELS_PATH, "popularity_model.csv"))

        logger.info(f"✅ Top {len(top_movies)} popular movies saved")  
        logger.info(f"Most popular: {top_movies.iloc[0]['title']}")

    return top_movies

def train_svd_model(train, test):  
    """Train SVD collaborative filtering model"""  
    logger.info("Training Model 2: SVD Collaborative Filtering...")

    reader = Reader(rating_scale=(0.5, 5.0))  
    data = Dataset.load_from_df(  
        train[['userId', 'movieId', 'rating']],  
        reader  
    )

    with mlflow.start_run(run_name="svd_model"):  
        params = {  
            'n_factors': 100,  
            'n_epochs': 20,  
            'lr_all': 0.005,  
            'reg_all': 0.02  
        }

        mlflow.log_params(params)  
        mlflow.log_param("model_type", "SVD")  
        mlflow.log_param("train_size", len(train))  
        mlflow.log_param("test_size", len(test))

        algo = SVD(  
            n_factors=params['n_factors'],  
            n_epochs=params['n_epochs'],  
            lr_all=params['lr_all'],  
            reg_all=params['reg_all']  
        )

        cv_results = cross_validate(  
            algo, data,  
            measures=['RMSE', 'MAE'],  
            cv=5,  
            verbose=False  
        )

        rmse = cv_results['test_rmse'].mean()  
        mae = cv_results['test_mae'].mean()

        mlflow.log_metric("rmse", round(rmse, 4))  
        mlflow.log_metric("mae", round(mae, 4))

        logger.info(f"✅ SVD RMSE: {rmse:.4f}")  
        logger.info(f"✅ SVD MAE: {mae:.4f}")

        trainset = data.build_full_trainset()  
        algo.fit(trainset)

        import pickle  
        model_path = os.path.join(MODELS_PATH, "svd_model.pkl")  
        with open(model_path, 'wb') as f:  
            pickle.dump(algo, f)

        mlflow.log_artifact(model_path)  
        logger.info(f"✅ SVD model saved to {model_path}")

    return algo, rmse, mae

def train_all():  
    """Main training function"""  
    logger.info("="*50)  
    logger.info("Starting Model Training...")  
    logger.info("="*50)

    os.makedirs(MODELS_PATH, exist_ok=True)

    mlflow.set_experiment("recommendation_system")

    train, test, item_popularity, movies = load_data()

    popularity_model = train_popularity_model(train, item_popularity)  
    svd_model, rmse, mae = train_svd_model(train, test)

    logger.info("="*50)  
    logger.info("Model Training Complete!")  
    logger.info(f"Model 1: Popularity-based (top 100 movies)")  
    logger.info(f"Model 2: SVD - RMSE: {rmse:.4f}, MAE: {mae:.4f}")  
    logger.info("="*50)

    return popularity_model, svd_model, rmse, mae

if __name__ == "__main__":  
    train_all()  
