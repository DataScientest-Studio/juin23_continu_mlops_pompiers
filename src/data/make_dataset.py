from import_raw_data import result, columns
import pandas as pd
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
  label_encoder = LabelEncoder()
  df_encoded = data[string_cols].apply(label_encoder.fit_transform)
  df_int = data.drop(string_cols, axis=1)

  df = df_int.join(df_encoded) #concaténation des variables integer et string encodés

  return df


def create_and_drop_columns(data):
    '''
    Créé la variable 'Month' au format int
    Supprime la variable DateOfCall
    '''
    data['Month'] = data['DateOfCall'].dt.month.astype(int)
    data.drop('DateOfCall', axis=1, inplace=True)

    return data


# Application des transformations :
data = load_data(result, columns)
data = convert_data_types(data)
df = create_and_drop_columns(data)

print(df.info())
