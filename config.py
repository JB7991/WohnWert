# Konfiguration: Städtenamen, PLZ, Koordinaten und Wirtschaftsdaten

APP_NAME = "WertWohn"
APP_UNTERTITEL = "Schweizer Immobilienpreisschätzer für Kauf & Miete"

# Schweizer Städte mit PLZ, Geokoordinaten, Kanton und Durchschnittseinkommen
STAEDTE = {
    "Zürich":            {"plz": "8000", "lat": 47.377, "lon": 8.542,  "kanton": "ZH", "einkommen": 95000},
    "Bern":              {"plz": "3000", "lat": 46.948, "lon": 7.447,  "kanton": "BE", "einkommen": 78000},
    "Basel":             {"plz": "4000", "lat": 47.560, "lon": 7.589,  "kanton": "BS", "einkommen": 82000},
    "Genf":              {"plz": "1201", "lat": 46.204, "lon": 6.143,  "kanton": "GE", "einkommen": 92000},
    "Lausanne":          {"plz": "1000", "lat": 46.520, "lon": 6.632,  "kanton": "VD", "einkommen": 80000},
    "Zug":               {"plz": "6300", "lat": 47.166, "lon": 8.516,  "kanton": "ZG", "einkommen": 105000},
    "Winterthur":        {"plz": "8400", "lat": 47.501, "lon": 8.724,  "kanton": "ZH", "einkommen": 72000},
    "Luzern":            {"plz": "6000", "lat": 47.050, "lon": 8.309,  "kanton": "LU", "einkommen": 74000},
    "St. Gallen":        {"plz": "9000", "lat": 47.425, "lon": 9.377,  "kanton": "SG", "einkommen": 68000},
    "Lugano":            {"plz": "6900", "lat": 46.004, "lon": 8.951,  "kanton": "TI", "einkommen": 70000},
}

# Alphabetisch sortierte Stadtliste für Dropdowns
STADTLISTE = sorted(STAEDTE.keys())

# Navigationsseiten der App
SEITEN = ["🏠 Preisschätzung", "📊 Marktübersicht", "📋 Meine Immobilien"]
