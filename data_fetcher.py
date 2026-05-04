# Nominatim API: Koordinaten für Schweizer Städte laden
import requests

def koordinaten_laden(stadtname):
    # Geokoordinaten über OpenStreetMap Nominatim API abrufen
    # Bei Fehler: None zurückgeben, gespeicherte Fallback-Daten bleiben erhalten
    try:
        antwort = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{stadtname}, Schweiz", "format": "json", "limit": 1},
            headers={"User-Agent": "WertWohn/1.0 (Universitaetsprojekt)"},
            timeout=4,
        )
        treffer = antwort.json()
        if treffer:
            lat = float(treffer[0]["lat"])
            lon = float(treffer[0]["lon"])
            return lat, lon
    except Exception:
        pass  # Bei jedem API-Fehler: Fallback auf gespeicherte Koordinaten
    return None

def koordinaten_alle_aktualisieren(stadtliste):
    # Koordinaten für alle Städte im Hintergrund über API laden
    import database
    for stadt in stadtliste:
        ergebnis = koordinaten_laden(stadt)
        if ergebnis:
            lat, lon = ergebnis
            database.koordinaten_aktualisieren(stadt, lat, lon)
