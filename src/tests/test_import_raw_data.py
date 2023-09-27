from data.import_raw_data import result, columns

import mysql.connector
import pytest

from decouple import config

# Chargez le fichier .env à partir du même répertoire que le script
config.config('.env')

DB_HOST = config('DB_HOST')
DB_USER = config('DB_USER')
DB_PASSWORD = config('DB_PASSWORD')
DB_NAME = config('DB_NAME')

# Vérifier la connexion à la base de donnée
def test_database_connection():
  try:
      connection = mysql.connector.connect(
        host='DB_HOST',
        user='DB_USER',
        password='DB_PASSWORD',
        database='DB_NAME'
      )
      assert connection.is_connected() == True
  except mysql.connector.Error as err:
      pytest.fail(f"Erreur de connexion à la base de données : {err}")
  finally:
      if connection.is_connected():
          connection.close()


# Vérifie le nombre de colonnes de la database
def test_len_columns():
  assert len(columns) == 14




