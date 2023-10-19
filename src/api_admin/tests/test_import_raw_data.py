from data.import_raw_data import columns

import mysql.connector
import pytest
from decouple import config

#login : 
admin = "admin"
password = config('PWD_DB_ADMIN', default='')

# Vérifier la connexion à la base de donnée
def test_database_connection():
  try:
      connection = mysql.connector.connect(
      host=config('DB_HOST'), # databse hébergée sur serveur AWS
      user=config('DB_USER'),
      password=config('DB_PASSWORD'),
      database=config('DB_NAME')
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




