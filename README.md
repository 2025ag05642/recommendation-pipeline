# Recommendation Pipeline

## End-to-End Data Management Pipeline for a Recommendation System

**Roll No: 2025AG05642 | DM4ML Assignment 2**

## Overview  
This project implements an end-to-end data management pipeline for a movie recommendation system using the MovieLens dataset.

## Pipeline Stages  
1. Data Ingestion 
2. Data Validation
3. Data Preparation
4. Data Transformation
5. Feature Management
6. Model Training
7. Model Evaluation
8. Orchestration (Apache Airflow)

## Dataset  
- Source: MovieLens Small Dataset 
- Ratings: 100,836 
- Movies: 9,742 
- Users: 610

## Models  
- Model 1: Popularity-based Recommender (Baseline) 
- Model 2: SVD Collaborative Filtering (Personalized)

## Evaluation Results (K=10)

| Metric       | Popularity Model | SVD Model | 
|--------------|------------------|-----------| 
| Precision@10 | 0.3086 	  | 0.1660 | 
| Recall@10    | 0.0524 	  | 0.0255 | 
| NDCG@10      | 0.3375 	  | 0.2063 | 
| Coverage     | 0.0013 	  | 0.0144 | 
| Diversity    | 1.0000 	  | 0.9458 | 
| RMSE         | N/A 	  	  | 0.8812 | 
| MAE          | N/A 		  | 0.6796 |

## Project Structure

    recommendation-pipeline/  
    ├── data/  
    │   ├── raw/            # MovieLens raw data  
    │   ├── processed/      # Cleaned and transformed data  
    │   ├── features/       # Engineered features  
    │   └── errors/         # Validation error logs  
    ├── src/  
    │   ├── ingestion/      # Data ingestion module  
    │   ├── validation/     # Data validation module  
    │   ├── preparation/    # Data preparation module  
    │   ├── transformation/ # Data transformation module  
    │   ├── features/       # Feature engineering module  
    │   ├── training/       # Model training module       
    │   └── evaluation/     # Model evaluation module  
    ├── dags/               # Apache Airflow DAG  
    ├── configs/            # Configuration files  
    ├── models/             # Trained models  
    ├── reports/            # Evaluation reports  
    ├── requirements.txt    # Python dependencies  
    └── README.md           # Project documentation

## Prerequisites  
- Python 3.12
- Apache Airflow 2.10.0
- Ubuntu/WSL2

## Installation Steps

Step 1: Clone repository

    git clone https://github.com/2025ag05642/recommendation-pipeline.git  
    cd recommendation-pipeline

Step 2: Create virtual environment

    python3.12 -m venv airflow_env  
    source airflow_env/bin/activate

Step 3: Install dependencies

    pip install -r requirements.txt

Step 4: Initialize Airflow

    export AIRFLOW_HOME=~/airflow  
    airflow db init

Step 5: Create Airflow admin user

    airflow users create --username admin --firstname Debabrata --lastname Santra --role Admin --email 2025ag05642@wilp.bits-pilani.ac.in --password admin123

## Download Dataset

    cd data/raw  
    wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip  
    unzip ml-latest-small.zip  
    mv ml-latest-small/* .

## Run Pipeline Manually

Run each stage individually:

    python3 src/ingestion/ingest.py  
    python3 src/validation/validate.py  
    python3 src/preparation/prepare.py  
    python3 src/transformation/transform.py  
    python3 src/features/feature_engineering.py  
    python3 src/training/train.py  
    python3 src/evaluation/evaluate.py

## Run via Apache Airflow

Step 1: Start Airflow webserver

    airflow webserver --port 8080 &

Step 2: Start Airflow scheduler

    airflow scheduler &

Step 3: Copy DAG to Airflow

    cp dags/recommendation_pipeline_dag.py ~/airflow/dags/

Step 4: Open Airflow UI

    Open browser: http://localhost:8080  
    Username: admin  
    Password: admin123

Step 5: Trigger pipeline

    Search for: recommendation_pipeline  
    Toggle DAG to ON  
    Click Trigger DAG button  
    Monitor all 7 tasks turning green

## Tools Used

| Tool | Purpose |  
|------|---------|  
| Python 3.12 | Core programming language |  
| Pandas | Data manipulation |  
| Pandera | Data validation |  
| Scikit-learn | Machine learning utilities |  
| Scikit-surprise | SVD recommendation model |  
| MLflow | Experiment tracking |  
| Apache Airflow 2.10.0 | Pipeline orchestration |  
| Parquet | Efficient data storage |

## Feature Catalog

| Feature Name | Type | Description |  
|-------------|------|-------------|  
| user_avg_rating | User | Average rating given by user |  
| user_activity_count | User | Total number of ratings by user |  
| user_std_rating | User | Standard deviation of user ratings |  
| days_since_last_interaction | User | Days since user last interacted |  
| preferred_category | User | Users most rated genre category |  
| item_avg_rating | Item | Average rating received by item |  
| item_popularity | Item | Total number of ratings for item |  
| item_std_rating | Item | Standard deviation of item ratings |

## Pipeline Performance  
- End-to-end runtime: 53 seconds  
- Train set: 80,668 ratings  
- Test set: 20,168 ratings  
- SVD RMSE: 0.8812  
- SVD MAE: 0.6796  
- Popularity Precision@10: 0.3086

## Data Validation Checks  
- Missing values check  
- Duplicate records check  
- Data type validation  
- Invalid ratings check (range 0.5-5.0)  
- Invalid timestamps check  
- Referential integrity check

## Author  
- Name: Debabrata Santra  
- Roll No: 2025AG05642  
- Course: Data Management for Machine Learning  
- Assignment: 2  # recommendation-pipeline
DMML Assignment 2
