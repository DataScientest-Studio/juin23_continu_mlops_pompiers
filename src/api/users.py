from fastapi import status,Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from decouple import config

# Fonction pour la vérification des identifiants et mots de passe des utilisateurs.
def verify_credentials(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    user = credentials.username
    password = credentials.password
    
    users_db = {
        "willy": config('PWD_DB_WILLY', default=''),
        "djamel": config('PWD_DB_DJAMEL', default=''),
        "jonathan": config('PWD_DB_JONATHAN', default=''),
        "root": config('PWD_DB_ROOT', default=''),
        "admin": config('PWD_DB_ADMIN', default=''),
    }

    if user in users_db and password == users_db[user]:
        return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )