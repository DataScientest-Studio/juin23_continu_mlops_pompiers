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