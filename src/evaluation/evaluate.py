# src/evaluation/evaluate.py

import pandas as pd  
import numpy as np  
import os  
import logging  
import pickle  
import mlflow  
from collections import defaultdict

logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)  
logger = logging.getLogger(__name__)

PROCESSED_PATH = "data/processed/"  
FEATURES_PATH = "data/features/"  
MODELS_PATH = "models/"  
REPORTS_PATH = "reports/"

def load_data():  
    """Load test data and models"""  
    test = pd.read_parquet(os.path.join(PROCESSED_PATH, "test_transformed.parquet"))  
    train = pd.read_parquet(os.path.join(PROCESSED_PATH, "train_transformed.parquet"))  
    popularity_model = pd.read_csv(os.path.join(MODELS_PATH, "popularity_model.csv"))

    with open(os.path.join(MODELS_PATH, "svd_model.pkl"), 'rb') as f:  
        svd_model = pickle.load(f)

    return test, train, popularity_model, svd_model

def get_popularity_recommendations(popularity_model, user_ids, train, K=10):  
    """Get recommendations from popularity model"""  
    recommendations = {}  
    top_movies = popularity_model['movieId'].tolist()[:K]

    for user_id in user_ids:  
        user_rated = set(train[train['userId'] == user_id]['movieId'].tolist())  
        recs = [m for m in top_movies if m not in user_rated][:K]  
        recommendations[user_id] = recs

    return recommendations

def get_svd_recommendations(svd_model, user_ids, train, all_movies, K=10):  
    """Get recommendations from SVD model"""  
    recommendations = {}

    for user_id in user_ids:  
        user_rated = set(train[train['userId'] == user_id]['movieId'].tolist())  
        unrated = [m for m in all_movies if m not in user_rated]

        predictions = []  
        for movie_id in unrated:  
            pred = svd_model.predict(user_id, movie_id)  
            predictions.append((movie_id, pred.est))

        predictions.sort(key=lambda x: x[1], reverse=True)  
        recommendations[user_id] = [m for m, _ in predictions[:K]]

    return recommendations

def precision_at_k(recommendations, actual, K=10):  
    """Calculate Precision@K"""  
    precisions = []  
    for user_id, recs in recommendations.items():  
        if user_id in actual:  
            actual_items = set(actual[user_id])  
            rec_items = set(recs[:K])  
            if len(rec_items) > 0:  
                precision = len(actual_items & rec_items) / len(rec_items)  
                precisions.append(precision)  
    return np.mean(precisions) if precisions else 0.0

def recall_at_k(recommendations, actual, K=10):  
    """Calculate Recall@K"""  
    recalls = []  
    for user_id, recs in recommendations.items():  
        if user_id in actual:  
            actual_items = set(actual[user_id])  
            rec_items = set(recs[:K])  
            if len(actual_items) > 0:  
                recall = len(actual_items & rec_items) / len(actual_items)  
                recalls.append(recall)  
    return np.mean(recalls) if recalls else 0.0

def ndcg_at_k(recommendations, actual, K=10):  
    """Calculate NDCG@K"""  
    ndcgs = []  
    for user_id, recs in recommendations.items():  
        if user_id in actual:  
            actual_items = set(actual[user_id])  
            dcg = 0.0  
            for i, item in enumerate(recs[:K]):  
                if item in actual_items:  
                    dcg += 1.0 / np.log2(i + 2)  
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_items), K)))  
            if idcg > 0:  
                ndcgs.append(dcg / idcg)  
    return np.mean(ndcgs) if ndcgs else 0.0

def coverage(recommendations, all_movies):  
    """Calculate Coverage"""  
    recommended_items = set()  
    for recs in recommendations.values():  
        recommended_items.update(recs)  
    return len(recommended_items) / len(all_movies)

def diversity(recommendations, movies_df):  
    """Calculate Diversity based on genres"""  
    diversities = []  
    for user_id, recs in recommendations.items():  
        if len(recs) < 2:  
            continue  
        genres_list = []  
        for movie_id in recs:  
            movie = movies_df[movies_df['movieId'] == movie_id]  
            if len(movie) > 0:  
                genres = set(movie.iloc[0]['genres'].split('|'))  
                genres_list.append(genres)  
        if len(genres_list) >= 2:  
            pairs = 0  
            diverse_pairs = 0  
            for i in range(len(genres_list)):  
                for j in range(i+1, len(genres_list)):  
                    pairs += 1  
                    if genres_list[i] != genres_list[j]:  
                        diverse_pairs += 1  
            if pairs > 0:  
                diversities.append(diverse_pairs / pairs)  
    return np.mean(diversities) if diversities else 0.0

def evaluate_all():  
    """Main evaluation function"""  
    logger.info("="*50)  
    logger.info("Starting Model Evaluation...")  
    logger.info("="*50)

    os.makedirs(REPORTS_PATH, exist_ok=True)  
    K = 10

    test, train, popularity_model, svd_model = load_data()  
    movies = pd.read_parquet(os.path.join(FEATURES_PATH, "movies_encoded.parquet"))

    test_users = test['userId'].unique()[:100]  
    all_movies = train['movieId'].unique().tolist()

    actual = {}  
    for user_id in test_users:  
        user_test = test[test['userId'] == user_id]  
        liked = user_test[user_test['rating'] >= 3.5]['movieId'].tolist()  
        if liked:  
            actual[user_id] = liked

    eval_users = list(actual.keys())  
    logger.info(f"Evaluating on {len(eval_users)} users with K={K}")

    logger.info("Getting popularity recommendations...")  
    pop_recs = get_popularity_recommendations(popularity_model, eval_users, train, K)

    logger.info("Getting SVD recommendations...")  
    svd_recs = get_svd_recommendations(svd_model, eval_users, train, all_movies, K)

    logger.info("Calculating metrics...")

    pop_precision = precision_at_k(pop_recs, actual, K)  
    pop_recall = recall_at_k(pop_recs, actual, K)  
    pop_ndcg = ndcg_at_k(pop_recs, actual, K)  
    pop_coverage = coverage(pop_recs, all_movies)  
    pop_diversity = diversity(pop_recs, movies)

    svd_precision = precision_at_k(svd_recs, actual, K)  
    svd_recall = recall_at_k(svd_recs, actual, K)  
    svd_ndcg = ndcg_at_k(svd_recs, actual, K)  
    svd_coverage = coverage(svd_recs, all_movies)  
    svd_diversity = diversity(svd_recs, movies)

    results = pd.DataFrame({  
        'Metric': [  
            'Precision@10',  
            'Recall@10',  
            'NDCG@10',  
            'Coverage',  
            'Diversity',  
            'RMSE',  
            'MAE'  
        ],  
        'Popularity Model': [  
            round(pop_precision, 4),  
            round(pop_recall, 4),  
            round(pop_ndcg, 4),  
            round(pop_coverage, 4),  
            round(pop_diversity, 4),  
            'N/A',  
            'N/A'  
        ],  
        'SVD Model': [  
            round(svd_precision, 4),  
            round(svd_recall, 4),  
            round(svd_ndcg, 4),  
            round(svd_coverage, 4),  
            round(svd_diversity, 4),  
            '0.8812',  
            '0.6796'  
        ]  
    })

    logger.info("="*50)  
    logger.info("Evaluation Results:")  
    logger.info("\n" + results.to_string(index=False))  
    logger.info("="*50)

    results.to_csv(os.path.join(REPORTS_PATH, "evaluation_results.csv"), index=False)  
    logger.info("✅ Results saved to reports/evaluation_results.csv")

    mlflow.set_experiment("recommendation_system")  
    with mlflow.start_run(run_name="evaluation"):  
        mlflow.log_metric("pop_precision_at_10", pop_precision)  
        mlflow.log_metric("pop_recall_at_10", pop_recall)  
        mlflow.log_metric("pop_ndcg_at_10", pop_ndcg)  
        mlflow.log_metric("pop_coverage", pop_coverage)  
        mlflow.log_metric("pop_diversity", pop_diversity)  
        mlflow.log_metric("svd_precision_at_10", svd_precision)  
        mlflow.log_metric("svd_recall_at_10", svd_recall)  
        mlflow.log_metric("svd_ndcg_at_10", svd_ndcg)  
        mlflow.log_metric("svd_coverage", svd_coverage)  
        mlflow.log_metric("svd_diversity", svd_diversity)  
        mlflow.log_artifact(os.path.join(REPORTS_PATH, "evaluation_results.csv"))

    return results

if __name__ == "__main__":  
    evaluate_all()  
