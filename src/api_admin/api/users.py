from fastapi import status,Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from decouple import config


#Dictionnaire des identifiants et des mots de passe :
users_db = {
        "willy": config('PWD_DB_WILLY', default=''),
        "djamel": config('PWD_DB_DJAMEL', default=''),
        "jonathan": config('PWD_DB_JONATHAN', default=''),
        "root": config('PWD_DB_ROOT', default=''),
        "admin": config('PWD_DB_ADMIN', default=''),
    }

# Fonction pour la vérification des identifiants et mots de passe des utilisateurs.
def verify_credentials(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    user = credentials.username
    password = credentials.password
    

    if not user or not password:
        raise HTTPException(
            status_code=401,
            detail="Veuillez vous identifier en fournissant un identifiant et un mot de passe valides",
            headers={"WWW-Authenticate": "Basic"},
        )
   
    if user in users_db:
        stored_password = users_db[user]
        if password == stored_password:
            return user
        else:
            raise HTTPException(
                status_code=401,
                detail="Non autorisé. Mauvais mot de passe",
                headers={"WWW-Authenticate": "Basic"},
            )
    else:
        raise HTTPException(
            status_code=401,
            detail="Non Autorisé. Mauvais identifiant",
            headers={"WWW-Authenticate": "Basic"},
        )
        
        
    
# Fonction pour la vérification de l'identifiant et mot de passe de l'administrateur.
def verify_credentials_admin(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    user = credentials.username
    password = credentials.password

    if not user or not password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Veuillez vous identifier en fournissant un identifiant et un mot de passe valides",
            headers={"WWW-Authenticate": "Basic"},
        )

    if user in users_db:
        stored_password = users_db[user]
        if password == stored_password:
            if user == "admin":  # Vérification de l'utilisateur admin
                return user
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Accès refusé. Vous devez être un administrateur.",
                    headers={"WWW-Authenticate": "Basic"},
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Non autorisé. Mauvais mot de passe",
                headers={"WWW-Authenticate": "Basic"},
            )
    
    else:
        return user