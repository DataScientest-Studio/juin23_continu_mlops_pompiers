from fastapi import status,Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from decouple import config



#Dictionnaire des identifiants et des mots de passe :
users_db = {
    "willy": "Pompiers2023*",
    "djamel": "pompiers",
    "jonathan": "pompiers",
    "root": "pompiers",
    "admin": "pompiers"
}

# Fonction pour la vérification des identifiants et mots de passe des utilisateurs.
def verify_credentials(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    user = credentials.username
    password = credentials.password
    
    # Chargez les identifiants et les mots de passe à partir des variables d'environnement
    users_db = {
        "willy": config('USER_DB_WILLY', default=''),
        "djamel": config('USER_DB_DJAMEL', default=''),
        "jonathan": config('USER_DB_JONATHAN', default=''),
        "root": config('USER_DB_ROOT', default=''),
        "admin": config('USER_DB_ADMIN', default=''),
    }

    if user in users_db and password == users_db[user]:
        return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )