from geopy.geocoders import Photon

geolocator = Photon()
addr = "123 Main Street, London, United Kingdom"
location = geolocator.geocode(addr)

print(location)

print(location.raw)