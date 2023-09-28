from data.import_raw_data import result, columns
from data.make_dataset import load_data
from models_training.model import r2_lgb, rmse_lgb
from decouple import config

from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import FastAPI, Depends, HTTPException, status
import uvicorn

#login : 
admin = "admin"
password = config('PWD_DB_ADMIN', default='')

security = HTTPBasic()


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


# Chargement de la base de donnée et conversion au format dictionnaire :
data_db = load_data(result, columns).to_dict(orient='records')

#Dictionnaire des codes d'erreur : 
responses = {
    200: {"description": "OK"},
    401: {"description" : "Veuillez vous identifier"},
    404: {"description": "Objet introuvable"},
    403: {"description": "Accès restreint"},
    406: {"description": "Mauvaise requête"}
}

@api.get('/', tags=['Home'], name='Welcome', responses=responses)
async def get_index(credentials: HTTPBasicCredentials = Depends(security)):
    """ Message de bienvenue
    """
    if credentials.username == admin and credentials.password == password:
        return {'message': f"Bonjour admin. Bienvenue sur l'API du projet London Fire Brigade"}
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès interdit. Vous devez être administrateur pour accéder à cette ressource.",
        )

@api.get('/data/columns', tags=['DataBase'], name='All Columns')
async def get_columns(credentials: HTTPBasicCredentials = Depends(security)):
    """Obtenir les colonnes du dataset"""
    return columns

@api.get('/data/sample', tags=['DataBase'], name='Sample')
async def get_sample(credentials: HTTPBasicCredentials = Depends(security)):
    """Obtenir les 20 dernières lignes de la base de donnée"""
    return data_db[-10:]

@api.get('/model/metrics/r2', tags=['Machine Learning'], name='Metrics R-squarred')
async def get_metrics_r2(credentials: HTTPBasicCredentials = Depends(security)):
    """Obtenir le score d'évaluation r² du modèle"""
    return f"R-squared (R²): {r2_lgb}"

@api.get('/model/metrics/rmse', tags=['Machine Learning'], name='Metrics RMSE')
async def get_metrics_rmse(credentials: HTTPBasicCredentials = Depends(security)):
    """Obtenir le score d'évaluation RMSE du modèle"""
    return f"Root Mean Squared Error (RMSE): {rmse_lgb}"


if __name__ == "__main__":
    uvicorn.run(api, host="127.0.0.1", port=8000)