# Wechselkurs API: Aktuelle EUR und USD Kurse für CHF laden
import requests
import pandas as pd

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
def hypothekarzinsen_holen():
    # Aktuelle Hypothekarzinsen vom SNB-Datenportal laden (Cube: zikrepro)
    url = "https://data.snb.ch/api/cube/zikrepro/data/json/de"
    try:
        antwort = requests.get(url, timeout=6)
        rohdaten = antwort.json()

        # SNB JSON-Struktur: Liste von [datum, serie, wert]
        eintraege = rohdaten.get("data", [])
        df = pd.DataFrame(eintraege, columns=["datum", "serie", "wert"])

        # Nur neueste Werte pro Serie behalten
        df["datum"] = pd.to_datetime(df["datum"])
        df = df.sort_values("datum").groupby("serie").last().reset_index()

        # Nur relevante Hypothekar-Serien (Festhypotheken + SARON)
        hypo_serien = {
            "ZIK1J": "Festhypothek 1 Jahr",
            "ZIK2J": "Festhypothek 2 Jahre",
            "ZIK3J": "Festhypothek 3 Jahre",
            "ZIK5J": "Festhypothek 5 Jahre",
            "ZIK10J": "Festhypothek 10 Jahre",
            "ZIKSARON": "SARON-Hypothek",
        }
        df = df[df["serie"].isin(hypo_serien.keys())].copy()
        df["bezeichnung"] = df["serie"].map(hypo_serien)
        df["wert"] = pd.to_numeric(df["wert"], errors="coerce")
        return df[["bezeichnung", "wert", "datum"]].sort_values("wert")

    except Exception:
        # Fallback auf fixe Richtwerte (Stand Mai 2026)
        import pandas as pd
        return pd.DataFrame({
            "bezeichnung": ["SARON-Hypothek", "Festhypothek 2 Jahre",
                            "Festhypothek 5 Jahre", "Festhypothek 10 Jahre"],
            "wert": [0.05, 0.85, 1.35, 1.75],
            "datum": [None] * 4,
        })
