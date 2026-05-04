# Hilfsfunktionen für WertWohn
from config import STAEDTE

def chf(betrag):
    # Betrag als CHF-String mit Tausendertrennzeichen formatieren
    return "CHF {:,.0f}".format(betrag).replace(",", "'")

def typ_bezeichnung(typ):
    # Internen Typ in lesbaren Text umwandeln
    if typ == "kauf":
        return "Kaufpreis"
    return "Monatliche Miete"

def suche_stadt(eingabe, stadtliste):
    # PLZ oder Stadtname in Stadtname aus der Liste umwandeln
    eingabe = eingabe.strip()
    if not eingabe:
        return None
    # 1. Exakte Übereinstimmung (Gross-/Kleinschreibung ignorieren)
    for name in stadtliste:
        if eingabe.lower() == name.lower():
            return name
    # 2. PLZ vergleichen
    for name, info in STAEDTE.items():
        if info.get("plz", "") == eingabe:
            return name
    # 3. Teilübereinstimmung im Stadtnamen
    for name in stadtliste:
        if eingabe.lower() in name.lower():
            return name
    return None
