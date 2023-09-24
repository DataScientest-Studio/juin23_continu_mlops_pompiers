from src.data.import_raw_data import result, columns, DB_HOST, DB_NAME, DB_PASSWORD, DB_USER

import mysql.connector
import pytest


# Vérifier la connexion à la base de donnée
def test_database_connection():
  try:    
      connection = mysql.connector.connect(
          host=DB_HOST,
          user=DB_USER,
          password=DB_PASSWORD,
          database=DB_NAME
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







# Vérifier la connexion à la base de donnée
def test_database_connection():
  try:
      connection = mysql.connector.connect(
          host=DB_HOST,
          user=DB_USER,
          password=DB_PASSWORD,
          database=DB_NAME
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
