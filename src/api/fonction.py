# Fonction pour formater le temps en minutes (arrondi à la minute supérieure)
def format_time(minutes: int) -> str:
    if minutes == 1:
        minute_str = "1 min"
    else:
        minute_str = f"{minutes} mins"

    return minute_str

def format_real_time(minutes: int, seconds: int) -> str:
    minute_str = "1 min" if minutes == 1 else f"{minutes} mins"
    second_str = "1 sec" if seconds == 1 else f"{seconds} secs"

    if minutes == 0:
        return second_str
    elif seconds == 0:
        return minute_str
    else:
        return f"{minute_str} {second_str}"
    
    
import jwt
from datetime import datetime, timedelta
import secrets

# Clé secrète pour signer le token (changez ceci en une clé sécurisée)
secret_key = secrets.token_hex(32)

# Fonction pour générer un token d'authentification
def generate_auth_token(username):
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(minutes=30)  # Expiration du token (ajustez selon vos besoins)
    }
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return token

