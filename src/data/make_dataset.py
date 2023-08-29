from import_raw_data import result, columns
import pandas as pd
from sklearn.preprocessing import LabelEncoder


data = pd.DataFrame(result, columns=columns)

# CONVERTIR LES TYPES DE DONNEES AU FORMAT APPROPRIE

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
data.drop(string_cols, axis=1, inplace=True)

df = data.join(df_encoded)


## Création/suppression de variables
# Extraire de 'DateofCall' une variable significative: le mois
df['Month'] = df['DateOfCall'].dt.month

# Suppression de la variable 'DateofCall'
df.drop('DateOfCall',axis=1, inplace=True)

print(df.info())
print(data.dtypes)
