from fastapi import status,Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from decouple import config

# Fonction pour la vérification des identifiants et mots de passe des utilisateurs.
def verify_credentials(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    user = credentials.username
    password = credentials.password
    
    # Chargez les identifiants et les mots de passe à partir des variables d'environnement
    users_db = {
        "willy": config('USER_DB_WILLY'),
        "djamel": config('USER_DB_DJAMEL'),
        "jonathan": config('USER_DB_JONATHAN'),
        "root": config('USER_DB_ROOT'),
        "admin": config('USER_DB_ADMIN'),
    }

    if user in users_db and password == users_db[user]:
        return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )