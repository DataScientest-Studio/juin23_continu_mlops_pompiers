import mysql.connector

# Connexion à la base de données
connection = mysql.connector.connect(
    host="lfb-project-db.cxwvi9sp2ptx.eu-north-1.rds.amazonaws.com", # databse hébergée sur serveur AWS
    user="admin",
    password="pompiers",
    database="london_fire_brigade"
)

# Création d'un curseur pour exécuter les requêtes
cursor = connection.cursor()

# Requête DELETE pour supprimer les lignes où IncGeo_WardCode ou IncidentStationGround est vide
delete_query = """
DELETE FROM incident
WHERE IncGeo_WardCode = '' OR IncidentStationGround = '';
"""

# Exécution de la requête DELETE
cursor.execute(delete_query)
connection.commit()

# Requête SELECT pour récupérer les données
select_query = """
SELECT 
    i.DateOfCall, i.HourOfCall, i.IncGeo_BoroughCode, i.IncGeo_WardCode, 
    i.Easting_rounded, i.Northing_rounded, i.IncidentStationGround,
    i.NumStationsWithPumpsAttending, i.NumPumpsAttending, i.PumpCount, i.PumpHoursRoundUp,
    m.PumpOrder, COALESCE(NULLIF(m.DelayCodeId, ''), 1) AS DelayCodeId, m.AttendanceTimeSeconds
FROM incident i
RIGHT JOIN mobilisation m ON i.IncidentNumber = m.IncidentNumber
WHERE i.DateOfCall IS NOT NULL AND i.PumpHoursRoundUp IS NOT NULL
"""

# Exécution de la requête SELECT
cursor.execute(select_query)

columns = [column[0] for column in cursor.description]

# Récupération des données
result = cursor.fetchall()

# Fermeture du curseur et de la connexion
cursor.close()
connection.close()
