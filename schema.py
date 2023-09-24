from pydantic import BaseModel



# Classe new_call pour faire une prédiction.
class NewCall(BaseModel):
    HourOfCall: int
    Easting_rounded: int
    Northing_rounded: int
    NumStationsWithPumpsAttending: int
    NumPumpsAttending: int
    PumpCount: int
    PumpHoursRoundUp: int
    PumpOrder: int
    DelayCodeId: int
    AttendanceTimeSeconds: int
    IncGeo_BoroughCode: int
    IncGeo_WardCode: int
    IncidentStationGround: int
    Month: int