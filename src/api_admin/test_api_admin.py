import pytest
from fastapi.testclient import TestClient

from api_admin import api
from decouple import config

# Utilisez le client de test FastAPI
client = TestClient(api)

#login : 
admin = "admin"
password = config('PWD_DB_ADMIN', default='')

def test_access_allowed():
    # Fonction de test pour vérifier l'accès autorisé
    response = client.get("/", auth=(admin, password))
    assert response.status_code == 200
    assert response.json() == {"message": "Bonjour admin. Bienvenue sur l'API du projet London Fire Brigade"}


def test_access_denied():
    # Fonction de test pour vérifier l'accès non autorisé
    response = client.get("/")
    assert response.status_code == 401

def test_get_sample():
    # Test du point de terminaison `/data/sample`
    response = client.get("/data/sample", auth=(admin, password))
    assert response.status_code == 200
    assert len(response.json()) == 10


def test_get_columns():
    # Fonction de test pour vérifier la récupération des colonnes
    response = client.get("/data/columns", auth=(admin, password))
    assert response.status_code == 200
    assert len(response.json()) == 14

