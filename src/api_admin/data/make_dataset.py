from data.import_raw_data import result, columns
import pandas as pd
from joblib import dump
import boto3
from decouple import config
from sklearn.preprocessing import LabelEncoder



# Chargement des données dans un dataframe :
def load_data(result, columns) :
  '''
  Récupère les données de la base de donnée et les stock dans un dataframe Pandas. 
  '''
  data_db = pd.DataFrame(result, columns=columns)
  return data_db


# CONVERTIR LES TYPES DE DONNEES AU FORMAT APPROPRIE
def transform_dataframe(data_db):
  '''
  Converti les données pour n'obtenir que des variables numériques dans le dataframe. 
  '''
  working_dataframe = data_db

  # Format date
  working_dataframe['DateOfCall'] = pd.to_datetime(working_dataframe['DateOfCall'])

  # En extraire une variable 'Month' et supprimer la colonne 'DateOfCall'
  working_dataframe['Month'] = working_dataframe['DateOfCall'].dt.month.astype(int)
  working_dataframe = working_dataframe.drop('DateOfCall', axis=1)
  
  # Variables au format integer
  colonnes_a_convertir = ['HourOfCall', 'Easting_rounded', 'Northing_rounded',
                'NumStationsWithPumpsAttending', 'NumPumpsAttending', 'PumpCount',
                'PumpHoursRoundUp', 'PumpOrder', 'DelayCodeId', 'AttendanceTimeSeconds']
  working_dataframe[colonnes_a_convertir] = working_dataframe[colonnes_a_convertir].astype(int)
  
  return working_dataframe


def encode_dataframe(working_dataframe):
  '''
  Encode les colonnes de type chaine de caractère avec un LabelEncoder.
  Le LabelEncoder ajusté aux données est ensuite chargé dans le bucket S3 de Amazon AWS.
  '''

  # Encodage des variables qui ont des valeurs sous forme de chaines de caractère
  string_cols = ['IncGeo_BoroughCode', 'IncGeo_WardCode', 'IncidentStationGround']
  working_dataframe[string_cols] = working_dataframe[string_cols].astype(str)
  encoder = LabelEncoder()
  working_dataframe[string_cols] = working_dataframe[string_cols].apply(encoder.fit_transform)
  dump(encoder, 'models/label_encoder.joblib') # Enregistrer le label_encoder ajusté aux données d'entrainement.

  # Upload du fichier label_encoder vers AWS S3
  s3_client = boto3.client('s3',region_name=config('AWS_S3_REGION'), aws_access_key_id=config('AWS_ADMIN_KEY_ID'), aws_secret_access_key=config('AWS_ADMIN_KEY'))
  s3_client.upload_file(Filename='models/label_encoder.joblib', Bucket=config('AWS_S3_BUCKET_NAME'), Key='label_encoder.joblib')
  
  df = working_dataframe

  return df


# Application des transformations :
data_db = load_data(result, columns)
working_dataframe = transform_dataframe(data_db)
df = encode_dataframe(working_dataframe)

print(f"working_dataframe : \n {working_dataframe.head()}")
print(f"df : \n {df.head()}")

