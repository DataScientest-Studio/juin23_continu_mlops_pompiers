#!/bin/bash

# Utiliser une image de base officielle pour Python
FROM python:3.11

# Definir le dossier de travail dans le conteneur
WORKDIR /app

# Créer le répertoire 'models' pour enregistrer les fichiers .joblib :
RUN mkdir /app/models 

# Copier les fichiers de dépendance dans le conteneur
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code source de l'application dans le conteneur
COPY . .

ENV PYTHONPATH=/app/src

# Exposer le port que l'API utilisera
EXPOSE 8001

# Commande à exécuter pour démarrer l'application
CMD ["uvicorn", "api_user:app", "--host", "0.0.0.0", "--port", "8001"]