from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import sys, os
import datetime as dt

from fuctions.update_model import update_model

ls_emails = ['carloalbertobarranco@gmail.com']

default_args = {
    'owner': 'Carlo',
    'depends_on_past': False,
    'email': ls_emails,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'update_model',
    default_args=default_args,
    description='Run update_model script',
    schedule_interval='0 7 * * *',
    start_date=days_ago(1)
) as dag:
    task_update_model = PythonOperator(
        task_id='update_model',
        python_callable=update_model,
    )
    task_update_model