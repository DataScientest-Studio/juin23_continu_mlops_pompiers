from import_raw_data import result, columns
from model import model_lgb, r2_lgb, rmse_lgb, scaler
from make_dataset import data

from fastapi import FastAPI, Header, HTTPException, Query, status, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

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
    "admin": "pompiers"
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
    HourOfCall: int #TODO Ajouter la condition pour que ce soit un entier compris entre 0 et 23
    IncGeo_BoroughCode: str
    IncGeo_WardCode: str
    Easting_rounded: int
    Northing_rounded: int
    IncidentStationGround: str
    NumStationsWithPumpsAttending: int
    NumPumpsAttending: int
    PumpCount: int
    PumpHoursRoundUp: int
    PumpOrder: int
    DelayCodeId: int
    Month: int #TODO Ajouter la condition pour que ce soit un entier compris entre 1 et 12

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
async def get_metrics_r2(current_user: str = Depends(verify_credentials)):
    """Obtenir le score d'évaluation r² du modèle"""
    return f"R-squared (R²): {r2_lgb}"

@api.get('/modele/metrics/rmse', tags=['Machine Learning'], name='Metrics RMSE')
async def get_metrics_rmse(current_user: str = Depends(verify_credentials)):
    """Obtenir le score d'évaluation RMSE du modèle"""
    return f"Root Mean Squared Error (RMSE): {rmse_lgb}"


@api.post("/modele/predict/", tags=['Machine Learning'], name='Prediction')
async def predict(new_call: NewCall):
    # Extraire les données
    input_data = pd.DataFrame([new_call.model_dump()])
    
    # Convertir les données str en int en utilisant le dictionnaire importé
    string_cols = ['IncGeo_BoroughCode', 'IncGeo_WardCode', 'IncidentStationGround']

    encoder = joblib.load('label_encoder.pkl') # Je charge le LabelEncoder déjà ajusté aux données d'entrainement

    input_data[string_cols] = input_data[string_cols].apply(encoder.fit_transform)

    # Standardiser les données
    scaled_data = scaler.transform(input_data)

    # Faire une prédiction à partir du modèle :
    prediction = model_lgb.predict(scaled_data)  # Adjust this based on your model's input format

    # Retourner la prédiction
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8000)

