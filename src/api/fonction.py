# Fonction pour formater le temps en minutes (arrondi à la minute supérieure)
def format_time(minutes: int) -> str:
    if minutes == 1:
        minute_str = "1 min"
    else:
        minute_str = f"{minutes} mins"

    return minute_str