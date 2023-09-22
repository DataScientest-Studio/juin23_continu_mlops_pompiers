from data.import_raw_data import result, columns
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


# Chargement des données dans un dataframe :
def load_data(result, columns) :
  '''
  Récupère les données de la base de donnée et les stock dans un dataframe Pandas. 
  '''
  data = pd.DataFrame(result, columns=columns)
  return data


# CONVERTIR LES TYPES DE DONNEES AU FORMAT APPROPRIE
def convert_data_types(data):
  '''
  Converti les données pour n'obtenir que des variables numériques dans le dataframe. 
  '''

  # Format date
  data['DateOfCall'] = pd.to_datetime(data['DateOfCall']) # Format date

  # Variables au format integer
  int_columns = ['HourOfCall', 'Easting_rounded', 'Northing_rounded',
                'NumStationsWithPumpsAttending', 'NumPumpsAttending', 'PumpCount',
                'PumpHoursRoundUp', 'PumpOrder', 'DelayCodeId', 'AttendanceTimeSeconds']

  data[int_columns] = data[int_columns].astype(int)

  # Encodage des variables qui ont des valeurs sous forme de chaines de caractère
  string_cols = ['IncGeo_BoroughCode', 'IncGeo_WardCode', 'IncidentStationGround']
  data[string_cols] = data[string_cols].astype(str)
  encoder = LabelEncoder()
  data[string_cols] = data[string_cols].apply(encoder.fit_transform)
  joblib.dump(encoder, 'label_encoder.pkl') # Enregistrer le label_encoder ajusté aux données d'entrainement.
  return data

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

print(df.info())
