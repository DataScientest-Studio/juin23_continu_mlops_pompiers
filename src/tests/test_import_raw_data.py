from data.import_raw_data import result, columns

import mysql.connector
import pytest

from decouple import config


DB_HOST = config('DB_HOST', default='')
DB_USER = config('DB_USER', default='')
DB_PASSWORD = config('DB_PASSWORD', default='')
DB_NAME = config('DB_NAME', default='')

# Vérifier la connexion à la base de donnée
def test_database_connection():
  try:
      connection = mysql.connector.connect(
        host='lfb-project-db.cxwvi9sp2ptx.eu-north-1.rds.amazonaws.com',
        user='admin',
        password='pompiers',
        database='london_fire_brigade'
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




