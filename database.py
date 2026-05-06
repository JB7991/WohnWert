# Datenbank-Verwaltung mit SQLite für WertWohn
import sqlite3
import random
import pandas as pd
from config import STAEDTE

DB_DATEI = "wertwohn.db"

def verbindung():
    # Neue Datenbankverbindung zurückgeben
    return sqlite3.connect(DB_DATEI)

def initialisieren():
    # Tabellen erstellen und Startdaten einfügen
    conn = verbindung()
    c = conn.cursor()

    # Tabelle für Immobilien
    c.execute("""
        CREATE TABLE IF NOT EXISTS immobilien (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stadt TEXT, kanton TEXT, flaeche REAL, zimmer REAL,
            stockwerk INTEGER, parkplatz INTEGER, baujahr INTEGER,
            preis REAL, typ TEXT
        )
    """)

    # Tabelle für Stadtkoordinaten und Wirtschaftsdaten
    c.execute("""
        CREATE TABLE IF NOT EXISTS stadtdaten (
            stadt TEXT PRIMARY KEY, lat REAL, lon REAL,
            kanton TEXT, einkommen REAL
        )
    """)
    conn.commit()

    # Stadtdaten aus config.py einfügen (nur wenn nicht vorhanden)
    for name, d in STAEDTE.items():
        c.execute(
            "INSERT OR IGNORE INTO stadtdaten VALUES (?,?,?,?,?)",
            (name, d["lat"], d["lon"], d["kanton"], d["einkommen"]),
        )
    conn.commit()

    # Beispieldaten erstellen falls die Tabelle leer ist
    c.execute("DELETE FROM immobilien")
    conn.commit()
    _beispieldaten_erstellen(conn)
    conn.close()

# Basis-Kaufpreise und Mietpreise pro Stadt (synthetische Marktdaten)
_KAUF = {
    "Zürich": 1200000, "Genf": 1100000, "Zug": 1350000, "Basel": 920000,
    "Lausanne": 960000, "Bern": 850000, "Winterthur": 780000, "Luzern": 810000,
    "St. Gallen": 680000, "Lugano": 720000,
}
_MIETE = {
    "Zürich": 2600, "Genf": 2500, "Zug": 2900, "Basel": 2000, "Lausanne": 2200,
    "Bern": 1900, "Winterthur": 1700, "Luzern": 1800, "St. Gallen": 1500,
    "Lugano": 1600,
}

def _beispieldaten_erstellen(conn):
    # Synthetische aber realistische Immobiliendaten für alle Städte generieren
    random.seed(42)
    c = conn.cursor()
    for stadt, info in STAEDTE.items():
        for _ in range(3):
            z = random.choice([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            f = round(max(25, z * 22 + random.gauss(0, 12)), 1)
            s = random.randint(0, 8)
            p = random.randint(0, 1)
            b = random.randint(1950, 2023)
            alter = 2024 - b

            # Kaufpreis: Basispreis angepasst nach Fläche, Alter und Parkplatz
            k = _KAUF[stadt] * (f / 80) * (1 - alter * 0.003) * (1 + p * 0.05)
            k *= random.uniform(0.88, 1.12)
            c.execute("INSERT INTO immobilien VALUES (NULL,?,?,?,?,?,?,?,?,?)",
                      (stadt, info["kanton"], f, z, s, p, b, round(k, -3), "kauf"))

            # Mietpreis: analog berechnet
            m = _MIETE[stadt] * (f / 80) * (1 - alter * 0.001) * (1 + p * 0.04)
            m *= random.uniform(0.9, 1.1)
            c.execute("INSERT INTO immobilien VALUES (NULL,?,?,?,?,?,?,?,?,?)",
                      (stadt, info["kanton"], f, z, s, p, b, round(m, -1), "miete"))
    conn.commit()

def laden(typ=None):
    # Immobilien aus der Datenbank laden, optional nach Typ filtern
    conn = verbindung()
    if typ:
        df = pd.read_sql("SELECT * FROM immobilien WHERE typ=?", conn, params=(typ,))
    else:
        df = pd.read_sql("SELECT * FROM immobilien", conn)
    conn.close()
    return df

def stadtdaten_laden():
    # Stadtkoordinaten und Wirtschaftsdaten laden
    conn = verbindung()
    df = pd.read_sql("SELECT * FROM stadtdaten", conn)
    conn.close()
    return df

def einfuegen(stadt, flaeche, zimmer, stockwerk, parkplatz, baujahr, preis, typ):
    # Neue Immobilie in die Datenbank schreiben
    conn = verbindung()
    kanton = STAEDTE.get(stadt, {}).get("kanton", "")
    conn.execute(
        """INSERT INTO immobilien
           (stadt, kanton, flaeche, zimmer, stockwerk, parkplatz, baujahr, preis, typ)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (stadt, kanton, flaeche, zimmer, stockwerk, parkplatz, baujahr, preis, typ),
    )
    conn.commit()
    conn.close()


