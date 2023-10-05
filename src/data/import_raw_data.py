import mysql.connector
#from decouple import config

from dotenv import load_dotenv
load_dotenv()
import os

# Accédez aux variables d'environnement
DB_HOST = os.environ.get('DB_HOST')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_DATABASE = os.environ.get('DB_NAME')


# Utilisez ces valeurs dans votre connexion MySQL
connection = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_DATABASE
)

# Création d'un curseur pour exécuter la requête SQL
cursor = connection.cursor()

# Exécution de la requête SQL dans MySQL
query = """
SELECT 
    i.DateOfCall, i.HourOfCall, i.IncGeo_BoroughCode, i.IncGeo_WardCode, 
    i.Easting_rounded, i.Northing_rounded, i.IncidentStationGround,
    i.NumStationsWithPumpsAttending, i.NumPumpsAttending, i.PumpCount, i.PumpHoursRoundUp,
    m.PumpOrder, COALESCE(NULLIF(m.DelayCodeId, ''), 1) AS DelayCodeId, m.AttendanceTimeSeconds
FROM incident i
RIGHT JOIN mobilisation m ON i.IncidentNumber = m.IncidentNumber
WHERE i.DateOfCall IS NOT NULL AND i.PumpHoursRoundUp IS NOT NULL
"""
cursor.execute(query)

columns = [column[0] for column in cursor.description]

# Récupération des données
result = cursor.fetchall()

# Fermeture du curseur et de la connexion
cursor.close()
connection.close()
