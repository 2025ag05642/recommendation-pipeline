# dags/recommendation_pipeline_dag.py

from airflow import DAG  
from airflow.operators.python import PythonOperator  
from datetime import datetime, timedelta  
import sys  
import os

sys.path.insert(0, '/home/aiengineeerdeb/recommendation-pipeline')

from src.ingestion.ingest import ingest_all  
from src.validation.validate import validate_all  
from src.preparation.prepare import prepare_all  
from src.transformation.transform import transform_all  
from src.features.feature_engineering import engineer_features  
from src.training.train import train_all  
from src.evaluation.evaluate import evaluate_all

default_args = {  
    'owner': 'debabrata_santra',  
    'depends_on_past': False,  
    'start_date': datetime(2026, 1, 1),  
    'email_on_failure': False,  
    'email_on_retry': False,  
    'retries': 1,  
    'retry_delay': timedelta(minutes=5),  
}

dag = DAG(  
    'recommendation_pipeline',  
    default_args=default_args,  
    description='End-to-End Recommendation System Pipeline',  
    schedule_interval=timedelta(days=1),  
    catchup=False,  
    tags=['recommendation', 'ml', 'dm4ml']  
)

def run_ingestion(**context):  
    os.chdir('/home/aiengineeerdeb/recommendation-pipeline')  
    ingest_all()  
    return "Ingestion Complete"

def run_validation(**context):  
    os.chdir('/home/aiengineeerdeb/recommendation-pipeline')  
    validate_all()  
    return "Validation Complete"

def run_preparation(**context):  
    os.chdir('/home/aiengineeerdeb/recommendation-pipeline')  
    prepare_all()  
    return "Preparation Complete"

def run_transformation(**context):  
    os.chdir('/home/aiengineeerdeb/recommendation-pipeline')  
    transform_all()  
    return "Transformation Complete"

def run_feature_engineering(**context):  
    os.chdir('/home/aiengineeerdeb/recommendation-pipeline')  
    engineer_features()  
    return "Feature Engineering Complete"

def run_training(**context):  
    os.chdir('/home/aiengineeerdeb/recommendation-pipeline')  
    train_all()  
    return "Training Complete"

def run_evaluation(**context):  
    os.chdir('/home/aiengineeerdeb/recommendation-pipeline')  
    evaluate_all()  
    return "Evaluation Complete"

t1_ingestion = PythonOperator(  
    task_id='data_ingestion',  
    python_callable=run_ingestion,  
    dag=dag  
)

t2_validation = PythonOperator(  
    task_id='data_validation',  
    python_callable=run_validation,  
    dag=dag  
)

t3_preparation = PythonOperator(  
    task_id='data_preparation',  
    python_callable=run_preparation,  
    dag=dag  
)

t4_transformation = PythonOperator(  
    task_id='data_transformation',  
    python_callable=run_transformation,  
    dag=dag  
)

t5_features = PythonOperator(  
    task_id='feature_engineering',  
    python_callable=run_feature_engineering,  
    dag=dag  
)

t6_training = PythonOperator(  
    task_id='model_training',  
    python_callable=run_training,  
    dag=dag  
)

t7_evaluation = PythonOperator(  
    task_id='model_evaluation',  
    python_callable=run_evaluation,  
    dag=dag  
)

t1_ingestion >> t2_validation >> t3_preparation >> t4_transformation >> t5_features >> t6_training >> t7_evaluation  
