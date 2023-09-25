import pytest
import requests

# Vérification des fonctions de l'API

def test_prediction():
    # Envoyer une requête avec des données valides
    data = {
        "HourOfCall": 23,
        "IncGeo_BoroughCode": "E09000011",
        "IncGeo_WardCode": "E05014075",
        "Easting_rounded": 541350,
        "Northing_rounded": 177950,
        "IncidentStationGround": "East Greenwich",
        "NumStationsWithPumpsAttending": 1,
        "NumPumpsAttending": 2,
        "PumpCount": 2,
        "PumpHoursRoundUp": 1,
        "PumpOrder": 2,
        "DelayCodeId": 1,
        "Month": 6
    }
    response = requests.post('http://127.0.0.1:8001/predict/', json=data)

    # Vérifier que la réponse a un code 200 (OK)
    assert response.status_code == 200

def test_prediction_invalid_datatypes():
    # Envoyer une requête avec des données non valides
    data = {
        "HourOfCall": 23, 
        "IncGeo_BoroughCode": "E09000011",
        "IncGeo_WardCode": "E05014075",
        "Easting_rounded": 541350,
        "Northing_rounded": 177950,
        "IncidentStationGround": 10, # type non valide (int au lieu de string)
        "NumStationsWithPumpsAttending": 1,
        "NumPumpsAttending": 2,
        "PumpCount": 2,
        "PumpHoursRoundUp": 1,
        "PumpOrder": 2,
        "DelayCodeId": 1,
        "Month": 6
    }
    response = requests.post('http://127.0.0.1:8001/predict/', json=data)

    # Vérifier que la réponse a un code 200 (OK)
    assert response.status_code == 422