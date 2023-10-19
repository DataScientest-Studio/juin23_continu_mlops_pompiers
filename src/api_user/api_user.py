from fastapi import FastAPI, Depends, Response
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
from decouple import config
import boto3
import math

from data.working_dataframe import working_dataframe
from api.schema import NewCall
from api.users import verify_credentials
from api.fonction import format_time

from joblib import load

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

# Login USER pour download sur AWS S3 :
s3_client = boto3.client('s3',region_name=config('AWS_S3_REGION'), aws_access_key_id=config('AWS_USER_KEY_ID'), aws_secret_access_key=config('AWS_USER_KEY'))


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


@app.get('/', include_in_schema=False)
def check_api():
    return {"message": "L'API est fonctionnelle"}

@app.post('/bienvenue', tags=['Home'], name='Welcome', responses={401: responses["401"]})
async def get_index(current_user: str = Depends(verify_credentials)):
    """ 
    Message de bienvenue
    """
    return {'message': f"Bonjour {current_user}. Bienvenue sur le projet London Fire Brigade"}


@app.post('/predict', tags=['Machine Learning'], name='predictions')
async def predict(new_call: NewCall):
    """
    Obtenir une prédiction à partir de nouvelles données d'entrée.
    Les données d'entrée doivent être une instance de la class NewCall.
    """

    # Vérifiez si les valeurs de NewCall ne correspondent pas aux valeurs de working_dataframe
    invalid_values = {col: value for col, value in new_call.model_dump().items() if value not in working_dataframe[col].values}

    if invalid_values:
        error_message = f"Les valeurs suivantes ne sont pas valides : {', '.join(invalid_values)}"
        return JSONResponse(content={"error": "Mauvaise Requête", "detail": error_message}, status_code=400)

    # Download des fichiers entrainés depuis AWS S3
    s3_client.download_file(Bucket=config('AWS_S3_BUCKET_NAME'), Key='lightgbm/model_lgb.joblib', Filename='models/model_lgb.joblib')
    s3_client.download_file(Bucket=config('AWS_S3_BUCKET_NAME'), Key='label_encoder.joblib', Filename='models/label_encoder.joblib')
    s3_client.download_file(Bucket=config('AWS_S3_BUCKET_NAME'), Key='scaler_fitted.joblib', Filename='models/scaler_fitted.joblib')
    
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

    response_text = f"Response time : {prediction_in_seconds} seconds. ({formatted_time})"
        
    # Retourner la prédiction
    return Response(content=response_text, media_type="text/plain")


if __name__ == '__main__':    
    uvicorn.run(app, host='127.0.0.1', port=8001)

