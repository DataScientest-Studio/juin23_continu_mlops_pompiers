from fastapi import status,Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials



#Dictionnaire des identifiants et des mots de passe :
users_db = {
    "willy": "Pompiers2023*",
    "djamel": "pompiers",
    "root": "pompiers",
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