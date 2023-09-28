from pydantic import BaseModel



# Classe new_call pour faire une prédiction.
class NewCall(BaseModel):
    HourOfCall: int #TODO Ajouter la condition pour que ce soit un entier compris entre 0 et 23
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
    Month: int #TODO Ajouter la condition pour que ce soit un entier compris entre 1 et 12
    
    
# Créez une instance de NewCall avec l'heure actuelle et le mois actuel
from datetime import datetime
current_hour = datetime.now().hour
current_month = datetime.now().month

new_call = NewCall(
    HourOfCall=current_hour,
    IncGeo_BoroughCode="example",
    IncGeo_WardCode="example",
    Easting_rounded=123,
    Northing_rounded=456,
    IncidentStationGround="example",
    NumStationsWithPumpsAttending=1,
    NumPumpsAttending=2,
    PumpCount=3,
    PumpHoursRoundUp=4,
    PumpOrder=5,
    DelayCodeId=6,
    Month=current_month
)