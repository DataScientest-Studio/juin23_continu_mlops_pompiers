from fastapi import FastAPI, Header,  Query, Depends
import uvicorn

from schema import NewCall
from api.users import verify_credentials

import datetime


from joblib import load

loaded_model_lgb = load('model_lgb.joblib')

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
    """ Message de bienvenue
    """
    return {'message': f"Bonjour {current_user}. Bienvenue sur l'API du projet London Fire Brigade for users"}


@app.post('/predict', tags=['Machine Learning'], name='predictions')
async def predict(data : NewCall):
     

    new_data = [[
            data.HourOfCall,
            data.Easting_rounded,
            data.Northing_rounded,
            data.NumStationsWithPumpsAttending,
            data.NumPumpsAttending,
            data.PumpCount,
            data.PumpHoursRoundUp,
            data.PumpOrder,
            data.DelayCodeId,
            data.IncGeo_BoroughCode,
            data.IncGeo_WardCode,
            data.IncidentStationGround,
            data.Month
    ]]


#predictions
    predictions = loaded_model_lgb.predict(new_data)[0]

 # Return the prediction
    return {'predictions': predictions}


if __name__ == '__main__':    
    uvicorn.run(app, host='localhost', port=8001)

