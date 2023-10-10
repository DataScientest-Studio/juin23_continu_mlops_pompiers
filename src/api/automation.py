from geopy.geocoders import Photon

geolocator = Photon()
addr = "Caxton Street, London, United Kingdom"
location = geolocator.geocode(addr)

print(location)

print(location.raw)