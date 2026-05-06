# Wechselkurs API: Aktuelle EUR und USD Kurse für CHF laden
import requests

def wechselkurs_holen():
    # Aktuellen Wechselkurs von frankfurter.app API laden
    try:
        antwort = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": "CHF", "to": "EUR,USD"},
            timeout=4,
        )
        daten = antwort.json()
        eur = daten["rates"]["EUR"]
        usd = daten["rates"]["USD"]
        return eur, usd
    except Exception:
        # Fallback auf fixe Kurse wenn API nicht erreichbar
        return 1.02, 1.10
