from fastapi import FastAPI, Depends
import uvicorn
import pandas as pd

from api.schema import NewCall
from api.users import verify_credentials
from models_training.model import scaler
import datetime

from joblib import load

loaded_model_lgb = load('models/model_lgb.joblib') # Chargement du modèle entrainé
encoder = load('models/label_encoder.joblib') # Chargement du LabelEncoder ajusté aux données d'entrainement

app = FastAPI(
    title='London Fire Brigade for users',
    description='MLOps projects based on the subject "London Fire Brigade Response Time"',
    version="1.0.1",
    openapi_tags=[
    {
        'name': 'Home',
        'description': 'Basic Functions'
    },
      {
        'name': 'Machine Learning',
        'description': 'Functions dealing with the Prediction Model'
    }]
    )





#Dictionnaire des codes d'erreur : 
responses = {
    200: {"description": "OK"},
    401: {"description" : "Veuillez vous identifier"},
    404: {"description": "Objet introuvable"},
    403: {"description": "Accès restreint"},
    406: {"description": "Mauvaise requête"}
}


@app.get('/', tags=['Home'], name='Welcome', responses=responses)
async def get_index(current_user: str = Depends(verify_credentials)):
    """ 
    Message de bienvenue
    """
    return {'message': f"Bonjour {current_user}. Bienvenue sur l'API du projet London Fire Brigade for users"}


@app.post('/predict', tags=['Machine Learning'], name='predictions')
async def predict(new_call: NewCall):
    """
    Obtenir une prédiction à partir de nouvelles données d'entrée.
    Les données d'entrée doivent être une instance de la class NewCall.
    """
    # Extraire les données
    input_data = pd.DataFrame([new_call.model_dump()])
    
    # Convertir les données str en int en utilisant le dictionnaire importé
    string_cols = ['IncGeo_BoroughCode', 'IncGeo_WardCode', 'IncidentStationGround']

    # Encoder les variables 'string' :
    input_data[string_cols] = input_data[string_cols].apply(encoder.fit_transform)

    # Standardiser les données
    scaled_data = scaler.transform(input_data)

    # Faire une prédiction à partir du modèle :
    prediction = loaded_model_lgb.predict(scaled_data) 

    # Retourner la prédiction
    return {"prediction": prediction[0]}


if __name__ == '__main__':    
    uvicorn.run(app, host='127.0.0.1', port=8001)
