from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import mysql.connector
from joblib import dump
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prepare_data(df):
    '''
    Sépare les features de la variable cible
    Sépare les données en set d'entrainement et de test
    Normalisation des données sur les ensembles de train et de test
    '''
    X = df.drop('AttendanceTimeSeconds', axis=1)
    y = df['AttendanceTimeSeconds']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, sc


def pred_model(model, X_test):
    '''
    Obtenir les prédictions du modèle
    '''
    y_pred = model.predict(X_test)
    return y_pred


# Configuration du DAG
default_args = {
    'owner': 'MLOps Datascientest',
    'depends_on_past': False,
    'start_date': datetime(2023, 10, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}


dag = DAG(
    'ALL-model_training_workflow',
    default_args=default_args,
    description='DAG pour le workflow d\'entraînement de modèle',
    schedule_interval=timedelta(minutes=30),  # Exécution toutes les demi-heures
    catchup=False,
)


def load_raw_data(**kwargs):
    connection = mysql.connector.connect(
        host="lfb-project-db.cxwvi9sp2ptx.eu-north-1.rds.amazonaws.com", # databse hébergée sur serveur AWS
        user="admin",
        password="pompiers",
        database="london_fire_brigade"
    )

    # Création d'un curseur pour exécuter la requête SQL
    cursor = connection.cursor()

    # Exécution de la requête SQL dans MySQL
    query = """
    SELECT 
        i.DateOfCall, i.HourOfCall, i.IncGeo_BoroughCode, i.IncGeo_WardCode, 
        i.Easting_rounded, i.Northing_rounded, i.IncidentStationGround,
        i.NumStationsWithPumpsAttending, i.NumPumpsAttending, i.PumpCount, i.PumpHoursRoundUp,
        m.PumpOrder, COALESCE(NULLIF(m.DelayCodeId, ''), 1) AS DelayCodeId, m.AttendanceTimeSeconds
    FROM incident i
    RIGHT JOIN mobilisation m ON i.IncidentNumber = m.IncidentNumber
    WHERE i.DateOfCall IS NOT NULL AND i.PumpHoursRoundUp IS NOT NULL
    """
    cursor.execute(query)

    columns = [column[0] for column in cursor.description]

    # Récupération des données
    result = cursor.fetchall()

    # Fermeture du curseur et de la connexion
    cursor.close()
    connection.close()
    return {"columns": columns, "result": result}
    

load_raw_data_task = PythonOperator(
    task_id='load_raw_data',
    python_callable=load_raw_data,
    provide_context=True,
    dag=dag,
)


def create_dataset(ti, **kwargs):
    
    data_from_prev_task = ti.xcom_pull(task_ids='load_raw_data')
    result = data_from_prev_task['result']
    columns = data_from_prev_task['columns']

    # Chargement des données dans un dataframe :
    def load_data(result, columns) :
        '''
        Récupère les données de la base de donnée et les stock dans un dataframe Pandas. 
        '''
        data_db = pd.DataFrame(result, columns=columns)
        return data_db

    # CONVERTIR LES TYPES DE DONNEES AU FORMAT APPROPRIE
    def convert_data_types(data):
        '''
        Converti les données pour n'obtenir que des variables numériques dans le dataframe. 
        '''
        converted_data = data

        # Format date
        converted_data['DateOfCall'] = pd.to_datetime(converted_data['DateOfCall']) # Format date

        # Variables au format integer
        int_columns = ['HourOfCall', 'Easting_rounded', 'Northing_rounded',
                      'NumStationsWithPumpsAttending', 'NumPumpsAttending', 'PumpCount',
                      'PumpHoursRoundUp', 'PumpOrder', 'DelayCodeId', 'AttendanceTimeSeconds']

        converted_data[int_columns] = converted_data[int_columns].astype(int)

        # Encodage des variables qui ont des valeurs sous forme de chaines de caractère
        string_cols = ['IncGeo_BoroughCode', 'IncGeo_WardCode', 'IncidentStationGround']
        converted_data[string_cols] = converted_data[string_cols].astype(str)
        encoder = LabelEncoder()
        converted_data[string_cols] = converted_data[string_cols].apply(encoder.fit_transform)
        dump(encoder, '/app/models/label_encoder.joblib') # Enregistrer le label_encoder ajusté aux données d'entrainement.
        return converted_data

    def create_and_drop_columns(converted_data):
        '''
        Créé la variable 'Month' au format int
        Supprime la variable DateOfCall
        '''
        converted_data['Month'] = converted_data['DateOfCall'].dt.month.astype(int)
        df = converted_data.drop('DateOfCall', axis=1)
        return df

    # Application des transformations :
    data_db = load_data(result, columns)
    converted_data = convert_data_types(data_db)
    df = create_and_drop_columns(converted_data)
    
    ti.xcom_push(key='transformed_data', value=df.to_json(date_format='iso'))  # Utilisez .to_json() pour sérialiser le DataFrame
    
    return df


create_dataset_task = PythonOperator(
    task_id='create_dataset',
    python_callable=create_dataset,
    provide_context=True,
    dag=dag,
)


def train_models(ti, **kwargs):
    df_json = ti.xcom_pull(task_ids='create_dataset', key='transformed_data')
    df = pd.read_json(df_json)

    # Préparation des données :
    X = df.drop('AttendanceTimeSeconds', axis=1)
    y = df['AttendanceTimeSeconds']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrainement des modèles
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    }

    model_paths = {}
    for model_name, model in models.items():
        # Training
        model.fit(X_train, y_train)
        
        # Save the model
        model_path = f'/tmp/{model_name}.joblib'
        dump(model, model_path)
        model_paths[model_name] = model_path

    ti.xcom_push(key='model_paths', value=model_paths)
    ti.xcom_push(key='X_test', value=X_test.tolist())  # Storing as a list for simplicity
    ti.xcom_push(key='y_test', value=y_test.tolist())  # Storing as a list for simplicity


train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    provide_context=True,
    dag=dag,
)


def evaluate_models(ti, **kwargs):
    model_paths = ti.xcom_pull(task_ids='train_models', key='model_paths')
    X_test = np.array(ti.xcom_pull(task_ids='train_models', key='X_test'))
    y_test = np.array(ti.xcom_pull(task_ids='train_models', key='y_test'))

    model_metrics = {}
    for model_name, model_path in model_paths.items():
        model = joblib.load(model_path)
        
        # Prediction
        y_pred = model.predict(X_test)
        
        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        model_metrics[model_name] = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "RMSE": rmse
        }

    ti.xcom_push(key='model_metrics', value=model_metrics)


evaluate_models_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    provide_context=True,
    dag=dag,
)


def select_best_model(ti):
    model_metrics = ti.xcom_pull(task_ids='evaluate_models', key='model_metrics')
    
    best_model_name = min(model_metrics, key=lambda k: model_metrics[k]['RMSE'])
    ti.xcom_push(key='best_model_name', value=best_model_name)


select_best_model_task = PythonOperator(
    task_id='select_best_model',
    python_callable=select_best_model,
    provide_context=True,
    dag=dag,
)


def export_best_model(ti):
    best_model_name = ti.xcom_pull(task_ids='select_best_model', key='best_model_name')
    model_paths = ti.xcom_pull(task_ids='train_models', key='model_paths')
    best_model_path = model_paths[best_model_name]
    
    best_model = joblib.load(best_model_path)
    export_path = f'/app/models/{best_model_name}.joblib'
    joblib.dump(best_model, export_path)


export_best_model_task = PythonOperator(
    task_id='export_best_model',
    python_callable=export_best_model,
    provide_context=True,
    dag=dag,
)


load_raw_data_task >> create_dataset_task >> train_models_task >> evaluate_models_task >> select_best_model_task >> export_best_model_task
