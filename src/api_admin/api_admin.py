from data.import_raw_data import result, columns
from data.make_dataset import load_data, transform_dataframe, encode_dataframe
from models_training.model import evaluate_model, pred_model, train_lightgbm, prepare_data, scale_data, train_random_forest
from api.users import verify_credentials_admin
from joblib import dump, load
from decouple import config
import boto3
import random

from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import FastAPI, Depends, HTTPException, status
import uvicorn


api = FastAPI(
    title='London Fire Brigade for admins',
    description='MLOps projects based on the subject "London Fire Brigade Response Time"',
    version="1.0.1",
    openapi_tags=[
    {
        'name': 'Home',
        'description': 'Basic Functions'
    },
    {
        'name': 'DataBase',
        'description': 'Functions dealing with the database'
    },
    {
        'name': 'Machine Learning',
        'description': 'Functions dealing with the Prediction Model'
    }]
    )

# Login pour download ou upload sur AWS S3 :
s3_client = boto3.client('s3',region_name=config('AWS_S3_REGION'), aws_access_key_id=config('AWS_ADMIN_KEY_ID'), aws_secret_access_key=config('AWS_ADMIN_KEY'))

# Chargement de la base de donnée et conversion au format dictionnaire :
data_db = load_data(result, columns).to_dict(orient='records')


#Dictionnaire des codes d'erreur : 
responses = {
    "200": {"description": "Succès"},
    "400": {"description": "Mauvaise requête. Problème de données."},
    "401": {"description": "Veuillez vous connecter avec des identifiants valides "},
    "403": {"description": "Accès restreint. Seul l'administrateur est autorisé à se connecter."},
    "422": {"description": "Entité non traitable"},
    "500": {"description": "Erreur interne du serveur"}
}

@api.get('/', tags=['Home'], name='Welcome', 
         responses={200: responses["200"], 401: responses["401"], 403: responses["403"], 500: responses["500"]})
async def get_index(credentials: HTTPBasicCredentials = Depends(verify_credentials_admin)):
    """ Message de bienvenue
    """
    return {'message': f"Bonjour admin. Bienvenue sur l'API du projet London Fire Brigade"}

@api.get('/data/columns', tags=['DataBase'], name='All Columns', 
         responses={200: responses["200"], 401: responses["401"], 403: responses["403"], 500: responses["500"]})
async def get_columns(credentials: HTTPBasicCredentials = Depends(verify_credentials_admin)):
    """Obtenir les colonnes du dataset"""
    return columns

@api.get('/data/sample', tags=['DataBase'], name='Sample', responses=responses)
async def get_sample(credentials: HTTPBasicCredentials = Depends(verify_credentials_admin)):
    """Obtenir un sample aléatoire de 10 lignes de la base de donnée"""
    random_10 = random.sample(data_db, 10)
    return random_10


@api.get('/model/metrics/lgbm', tags=['Machine Learning'], name='Metrics LightGBM', 
         responses={200: responses["200"], 401: responses["401"], 403: responses["403"], 422: responses["422"], 500: responses["500"]})
async def get_metrics_lgbm(current_user: str = Depends(verify_credentials_admin)):
    """Obtenir les scores d'évaluation du modèle LightGBM"""
    
    s3_client.download_file(Bucket=config('AWS_S3_BUCKET_NAME'), Key='lightgbm/mae_lgb.joblib', Filename='models/mae_lgb.joblib')
    mae_lgb = load('models/mae_lgb.joblib')

    s3_client.download_file(Bucket=config('AWS_S3_BUCKET_NAME'), Key='lightgbm/r2_lgb.joblib', Filename='models/r2_lgb.joblib')
    r2_lgb = load('models/r2_lgb.joblib')

    s3_client.download_file(Bucket=config('AWS_S3_BUCKET_NAME'), Key='lightgbm/rmse_lgb.joblib', Filename='models/rmse_lgb.joblib')
    rmse_lgb = load('models/rmse_lgb.joblib')

    return f"Mean Absolute Error (MAE): {mae_lgb}", f"Root Mean Squared Error (RMSE): {rmse_lgb}", f"R-squared (r2) : {r2_lgb}"


@api.get('/model/metrics/rf', tags=['Machine Learning'], name='Metrics Random Forest', 
         responses={200: responses["200"], 401: responses["401"], 403: responses["403"], 422: responses["422"], 500: responses["500"]})
async def get_metrics_rf(current_user: str = Depends(verify_credentials_admin)):
    """Obtenir les scores d'évaluation du modèle Random Forest"""

    s3_client.download_file(Bucket=config('AWS_S3_BUCKET_NAME'), Key='randomforest/mae_rf.joblib', Filename='models/mae_rf.joblib')
    mae_rf = load('models/mae_rf.joblib')

    s3_client.download_file(Bucket=config('AWS_S3_BUCKET_NAME'), Key='randomforest/r2_rf.joblib', Filename='models/r2_rf.joblib')
    r2_rf = load('models/r2_rf.joblib')

    s3_client.download_file(Bucket=config('AWS_S3_BUCKET_NAME'), Key='randomforest/rmse_rf.joblib', Filename='models/rmse_rf.joblib')
    rmse_rf = load('models/rmse_rf.joblib')

    return f"Mean Absolute Error (MAE): {mae_rf}", f"Root Mean Squared Error (RMSE): {rmse_rf}", f"R-squared (r2) : {r2_rf}"


@api.get('/model/training/lgbm', tags=['Machine Learning'], name='Train model LightGBM', responses=responses)
async def get_train_lgbm(credentials: HTTPBasicCredentials = Depends(verify_credentials_admin)):
    """Entrainer un modèle LightGBM sur les données de la base"""

    # Création du dataframe :
    data_db_source = load_data(result, columns)
    working_dataframe = transform_dataframe(data_db_source)
    df = encode_dataframe(working_dataframe)

    # Préparation des données :
    X_train, X_test, y_train, y_test = prepare_data(df)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Entrainement et résultats du modèle Light GBM :
    model_lgb = train_lightgbm(X_train_scaled, y_train)

    # Enregistrement du modèle :
    dump(model_lgb, 'models/model_lgb.joblib')
    s3_client.upload_file(Filename='models/model_lgb.joblib', Bucket=config('AWS_S3_BUCKET_NAME'), Key='lightgbm/model_lgb.joblib')

    # Calcul des métriques : 
    y_pred_lgb = pred_model(model_lgb, X_test_scaled)
    mse_lgb, mae_lgb, r2_lgb, rmse_lgb = evaluate_model('Light GBM', y_test, y_pred_lgb)

    # Enregistrement des métriques :
    dump(mse_lgb, 'models/mse_lgb.joblib')
    s3_client.upload_file(Filename='models/mse_lgb.joblib', Bucket=config('AWS_S3_BUCKET_NAME'), Key='lightgbm/mse_lgb.joblib')
    dump(mae_lgb, 'models/mae_lgb.joblib')
    s3_client.upload_file(Filename='models/mae_lgb.joblib', Bucket=config('AWS_S3_BUCKET_NAME'), Key='lightgbm/mae_lgb.joblib')
    dump(r2_lgb, 'models/r2_lgb.joblib')
    s3_client.upload_file(Filename='models/r2_lgb.joblib', Bucket=config('AWS_S3_BUCKET_NAME'), Key='lightgbm/r2_lgb.joblib')
    dump(rmse_lgb, 'models/rmse_lgb.joblib')
    s3_client.upload_file(Filename='models/rmse_lgb.joblib', Bucket=config('AWS_S3_BUCKET_NAME'), Key='lightgbm/rmse_lgb.joblib')

    return "Le modèle LightGBM a été entrainé."

@api.get('/model/training/rf', tags=['Machine Learning'], name='Train model Random Forest', responses=responses)
async def get_train_rf(credentials: HTTPBasicCredentials = Depends(verify_credentials_admin)):
    """Entrainer un modèle RandomForest sur les données de la base"""

    # Création du dataframe :
    data_db_source = load_data(result, columns)
    working_dataframe = transform_dataframe(data_db_source)
    df = encode_dataframe(working_dataframe)

    # Préparation des données :
    X_train, X_test, y_train, y_test = prepare_data(df)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Entrainement et résultats du modèle Light GBM :
    model_rf = train_random_forest(X_train_scaled, y_train)

    # Enregistrement du modèle :
    dump(model_rf, 'models/model_rf.joblib')
    s3_client.upload_file(Filename='models/model_rf.joblib', Bucket=config('AWS_S3_BUCKET_NAME'), Key='randomforest/model_rf.joblib')

    # Calcul des métriques :
    y_pred_rf = pred_model(model_rf, X_test_scaled)
    mse_rf, mae_rf, r2_rf, rmse_rf = evaluate_model('Random Forest', y_test, y_pred_rf)

    # Enregistrement des métriques :
    dump(mse_rf, 'models/mse_rf.joblib')
    s3_client.upload_file(Filename='models/mse_rf.joblib', Bucket=config('AWS_S3_BUCKET_NAME'), Key='randomforest/mse_rf.joblib')
    dump(mae_rf, 'models/mae_rf.joblib')
    s3_client.upload_file(Filename='models/mae_rf.joblib', Bucket=config('AWS_S3_BUCKET_NAME'), Key='randomforest/mae_rf.joblib')
    dump(r2_rf, 'models/r2_rf.joblib')
    s3_client.upload_file(Filename='models/r2_rf.joblib', Bucket=config('AWS_S3_BUCKET_NAME'), Key='randomforest/r2_rf.joblib')
    dump(rmse_rf, 'models/rmse_rf.joblib')
    s3_client.upload_file(Filename='models/rmse_rf.joblib', Bucket=config('AWS_S3_BUCKET_NAME'), Key='randomforest/rmse_rf.joblib')

    return "Le modèle RandomForest a été entrainé."


if __name__ == "__main__":
    uvicorn.run(api, host="127.0.0.1", port=8000)

