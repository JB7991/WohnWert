# Machine Learning Modell: Preisschätzung mit Random Forest
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import database

# Globale Speicherung: Modelle und Encoder bleiben nach dem Training erhalten
_modelle = {}
_encoder = None

def _encoder_initialisieren(daten):
    # LabelEncoder für Stadtnamen einmalig aufsetzen
    global _encoder
    if _encoder is None:
        _encoder = LabelEncoder()
        _encoder.fit(daten["stadt"].unique())

def _features_erstellen(daten):
    # Eingabevektoren für das Modell aus den Rohdaten aufbauen
    _encoder_initialisieren(daten)
    df = daten.copy()
    df["stadt_nr"] = _encoder.transform(df["stadt"].values)
    df["alter"] = datetime.now().year - df["baujahr"]
    # Features: Stadt (codiert), Fläche, Zimmer, Stockwerk, Parkplatz, Gebäudealter
    return df[["stadt_nr", "flaeche", "zimmer", "stockwerk", "parkplatz", "alter"]]

def trainieren(typ):
    # Random Forest für Kauf- oder Mietpreise trainieren und speichern
    global _modelle
    daten = database.laden(typ=typ)
    if len(daten) < 10:
        return
    X = _features_erstellen(daten)
    y = daten["preis"].values
    modell = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modell.fit(X, y)
    _modelle[typ] = modell

def alle_trainieren():
    # Beide Modelle (Kauf + Miete) beim App-Start laden
    trainieren("kauf")
    trainieren("miete")

def schaetzen(stadt, flaeche, zimmer, stockwerk, parkplatz, baujahr, typ):
    # Preis für eine Immobilie schätzen und als gerundeten CHF-Betrag zurückgeben
    if typ not in _modelle:
        trainieren(typ)
    if typ not in _modelle:
        # Einfacher Fallback falls kein Modell vorhanden
        return round(flaeche * (8000 if typ == "kauf" else 20), -2)
    try:
        # Unbekannte Städte werden auf den ersten bekannten Wert abgebildet
        stadt_nr = _encoder.transform([stadt])[0]
    except Exception:
        stadt_nr = 0
    alter = datetime.now().year - baujahr
    X = np.array([[stadt_nr, flaeche, zimmer, stockwerk, int(parkplatz), alter]])
    preis = _modelle[typ].predict(X)[0]
    # Kaufpreise auf CHF 1'000, Mieten auf CHF 10 runden
    runden = -3 if typ == "kauf" else -1
    return round(preis, runden)
