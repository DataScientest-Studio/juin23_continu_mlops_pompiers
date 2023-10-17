from fastapi import FastAPI, Depends, Response
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import math

from data.working_dataframe import working_dataframe
from api.schema import NewCall
from api.users import verify_credentials
from api.fonction import format_time

from joblib import load

loaded_model_lgb = load('models/model_lgb.joblib') # Chargement du modèle entrainé
encoder = load('models/label_encoder.joblib') # Chargement du LabelEncoder ajusté aux données d'entrainement

#Dictionnaire des codes d'erreur : 
responses = {
    "400": {"description": "Mauvaise requête"},
    "401": {"description": "Non autorisé"},
    "403": {"description": "Interdit"},
    "404": {"description": "Non trouvé"},
    "422": {"description": "Entité non traitable"},
    "500": {"description": "Erreur interne du serveur"},
    "503": {"description": "Service indisponible"},
}

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





@app.get('/', include_in_schema=False)
def check_api():
    return {"message": "L'API est fonctionnelle"}


@app.get('/bienvenue', tags=['Home'], name='Welcome', responses={401: responses["401"]})
async def get_index(current_user: str = Depends(verify_credentials)):
    """ 
    Message de bienvenue
    """
    return {'message': f"Bonjour {current_user}. Bienvenue sur l'API du projet London Fire Brigade for users"}



@app.post('/predict', tags=['Machine Learning'], name='predictions', responses={400: responses["400"],401: responses["401"], 422: responses["422"]})
async def predict(new_call: NewCall, current_user: str = Depends(verify_credentials)):
    """
    Obtenir une prédiction à partir de nouvelles données d'entrée.
    Les données d'entrée doivent être une instance de la class NewCall.
    """
    
    # Vérifiez si les valeurs de NewCall ne correspondent pas aux valeurs de working_dataframe
    invalid_values = {col: value for col, value in new_call.model_dump().items() if value not in working_dataframe[col].values}
    
    if invalid_values:
        error_message = f"Les valeurs suivantes ne sont pas valides : {', '.join(invalid_values)}"
        return JSONResponse(content={"error": "Mauvaise Requête", "detail": error_message}, status_code=400)
        
    loaded_model_lgb = load('models/model_lgb.joblib') # Chargement du modèle entrainé
    encoder = load('models/label_encoder.joblib') # Chargement du LabelEncoder ajusté aux données d'entrainement
    scaler = load('models/scaler_fitted.joblib') # Chargement du MinMaxScaler ajusté aux données d'entrainement
    
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
    prediction_in_seconds = prediction[0]

    # Arrondir à la minute supérieure
    rounded_seconds = round(prediction_in_seconds)
    minutes = math.ceil(rounded_seconds / 60)

    # Formater le temps en "X mins" (minute supérieure)
    formatted_time = format_time(minutes)

    response_text = f"Response time : {formatted_time}"
        
    # Retourner la prédiction
    return Response(content=response_text, media_type="text/plain")

    
if __name__ == '__main__':    
    uvicorn.run(app, host='127.0.0.1', port=8001)
    
    