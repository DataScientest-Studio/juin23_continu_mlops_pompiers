import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

data_dir = os.path.join(project_root, 'data')
sys.path.append(data_dir)

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

model_dir = os.path.join(project_root, 'models')
sys.path.append(model_dir)


from data.import_raw_data import result, columns
from data.make_dataset import data
from model import r2_lgb, rmse_lgb

from fastapi import FastAPI, Query, Depends
import uvicorn

import pandas as pd
from typing import Optional, List, Dict

from users import verify_credentials

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

if __name__ == '__main__':    
    uvicorn.run(api, host='localhost', port=8000)