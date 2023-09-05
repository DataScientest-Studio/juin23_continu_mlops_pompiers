from import_raw_data import result, columns
from model import model_lgb, r2_lgb, rmse_lgb
from make_dataset import data

from fastapi import FastAPI, Header, HTTPException, Query, status, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import pandas as pd
from pydantic import BaseModel
from typing import Optional, List, Dict

api = FastAPI(
    title='London Fire Brigade',
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


#Dictionnaire des identifiants et des mots de passe :
users_db = {
    "willy": "Pompiers2023*",
    "djamel": "pompiers",
    "root": "pompiers",
}

# Fonction pour la vérification des identifiants et mots de passe des utilisateurs.
def verify_credentials(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    user = credentials.username
    password = credentials.password

    if user in users_db and password == users_db[user]:
        return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )


# Classe new_call pour faire une prédiction.
class NewCall(BaseModel):
    HourOfCall: int
    Easting_rounded: int
    Northing_rounded: int
    NumStationsWithPumpsAttending: int
    NumPumpsAttending: int
    PumpCount: int
    PumpHoursRoundUp: int
    PumpOrder: int
    DelayCodeId: int
    AttendanceTimeSeconds: int
    IncGeo_BoroughCode: int
    IncGeo_WardCode: int
    IncidentStationGround: int
    Month: int

# Mise en forme de la base de donnée depuis le dataframe :
data = data.to_dict(orient='records')


#Dictionnaire des codes d'erreur : 
responses = {
    200: {"description": "OK"},
    401: {"description" : "Veuillez vous identifier"},
    404: {"description": "Objet introuvable"},
    403: {"description": "Accès restreint"},
    406: {"description": "Mauvaise requête"}
}

@api.get('/', tags=['Home'], name='Welcome', responses=responses)
async def get_index(current_user: str = Depends(verify_credentials)):
    """ Message de bienvenue
    """
    return {'message': f"Bonjour {current_user}. Bienvenue sur l'API du projet London Fire Brigade"}

@api.get('/data/columns', tags=['DataBase'], name='All Columns')
async def get_columns(current_user: str = Depends(verify_credentials)):
    """Obtenir les colonnes du dataset"""
    return columns

@api.get('/data/sample', tags=['DataBase'], name='Sample')
async def get_sample(current_user: str = Depends(verify_credentials)):
    """Obtenir les 20 dernières lignes de la base de donnée"""
    return data[-10:]

@api.get('/modele/metrics/r2', tags=['Machine Learning'], name='Metrics R-squarred')
async def get_metrics(current_user: str = Depends(verify_credentials)):
    """Obtenir le score d'évaluation r² du modèle"""
    return f"R-squared (R²): {r2_lgb}"

@api.get('/modele/metrics/rmse', tags=['Machine Learning'], name='Metrics RMSE')
async def get_metrics(current_user: str = Depends(verify_credentials)):
    """Obtenir le score d'évaluation RMSE du modèle"""
    return f"Root Mean Squared Error (RMSE): {rmse_lgb}"

