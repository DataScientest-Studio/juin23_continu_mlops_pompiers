from data.import_raw_data import result, columns
import pandas as pd


### Créez un DataFrame en spécifiant les noms des colonnes et en attribuant le type de données approprié à chacune d'entre elles.

# Construire le dataframe
working_dataframe = pd.DataFrame(result)

# Remplacez les noms des colonnes existantes par les nouveaux noms
working_dataframe.columns = columns

# Convertir la colonne 'DateOfCall' au format date
working_dataframe['DateOfCall'] = pd.to_datetime(working_dataframe['DateOfCall'])

# En extraire une variable 'Month' et supprimer la colonne 'DateOfCall'
working_dataframe['Month'] = working_dataframe['DateOfCall'].dt.month.astype(int)
working_dataframe = working_dataframe.drop('DateOfCall', axis=1)

# Convertir en type de données entier (int) les colonnes requises (les restantes seront de type chaîne de caractères (string))
colonnes_a_convertir= ["HourOfCall", "Easting_rounded", "Northing_rounded", "NumStationsWithPumpsAttending", 
                       "NumPumpsAttending", "PumpCount","PumpHoursRoundUp", "PumpOrder", "DelayCodeId", "AttendanceTimeSeconds"]
working_dataframe[colonnes_a_convertir] = working_dataframe[colonnes_a_convertir].astype(int)