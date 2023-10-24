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
def convert_data_types(data):
  '''
  Converti les données pour n'obtenir que des variables numériques dans le dataframe. 
  '''
  converted_data = data

  # Format date
  converted_data['DateOfCall'] = pd.to_datetime(converted_data['DateOfCall']) # Format date

  # Variables au format integer
  int_columns = ['HourOfCall', 'Easting_rounded', 'Northing_rounded',
                'NumStationsWithPumpsAttending', 'NumPumpsAttending', 'PumpCount',
                'PumpHoursRoundUp', 'PumpOrder', 'DelayCodeId', 'AttendanceTimeSeconds']

  converted_data[int_columns] = converted_data[int_columns].astype(int)

  # Encodage des variables qui ont des valeurs sous forme de chaines de caractère
  string_cols = ['IncGeo_BoroughCode', 'IncGeo_WardCode', 'IncidentStationGround']
  converted_data[string_cols] = converted_data[string_cols].astype(str)
  encoder = LabelEncoder()
  converted_data[string_cols] = converted_data[string_cols].apply(encoder.fit_transform)
  dump(encoder, 'models/label_encoder.joblib') # Enregistrer le label_encoder ajusté aux données d'entrainement.

  # Upload du fichier enregistré vers AWS S3
  s3_client = boto3.client('s3',region_name=config('AWS_S3_REGION'), aws_access_key_id=config('AWS_ADMIN_KEY_ID'), aws_secret_access_key=config('AWS_ADMIN_KEY'))
  s3_client.upload_file(Filename='models/label_encoder.joblib', Bucket=config('AWS_S3_BUCKET_NAME'), Key='label_encoder.joblib')

  return converted_data

def create_and_drop_columns(converted_data):
    '''
    Créé la variable 'Month' au format int
    Supprime la variable DateOfCall
    '''
    converted_data['Month'] = converted_data['DateOfCall'].dt.month.astype(int)
    df = converted_data.drop('DateOfCall', axis=1)
    return df


# Application des transformations :
data_db = load_data(result, columns)
converted_data = convert_data_types(data_db)
df = create_and_drop_columns(converted_data)
