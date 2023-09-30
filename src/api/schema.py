from pydantic import BaseModel


# Créez une instance de NewCall avec l'heure actuelle et le mois actuel
from datetime import datetime
current_hour = int(datetime.now().hour)
current_month = int(datetime.now().month)



# Classe new_call pour faire une prédiction.
class NewCall(BaseModel):
    HourOfCall: int = current_hour 
    IncGeo_BoroughCode: str
    IncGeo_WardCode: str
    Easting_rounded: int
    Northing_rounded: int
    IncidentStationGround: str
    NumStationsWithPumpsAttending: int
    NumPumpsAttending: int
    PumpCount: int
    PumpHoursRoundUp: int
    PumpOrder: int
    DelayCodeId: int
    Month: int = current_month
    