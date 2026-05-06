# WertWohn – Schweizer Immobilienpreisschätzer (Hauptdatei)
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import database
import ml_model
import data_fetcher
from config import STAEDTE, STADTLISTE, SEITEN, APP_NAME, APP_UNTERTITEL
from utils import chf, typ_bezeichnung, suche_stadt

# ── Seiteneinstellungen ───────────────────────────────────────────────────────
st.set_page_config(page_title=APP_NAME, page_icon="🏠", layout="wide")

# ── Initialisierung beim ersten Start ─────────────────────────────────────────

if "gestartet" not in st.session_state:
    database.initialisieren()
    ml_model.alle_trainieren()
    st.session_state["gestartet"] = True
    st.cache_data.clear()

# ── Daten mit Cache laden ─────────────────────────────────────────────────────

@st.cache_data
def daten_holen(typ):
    return database.laden(typ=typ)

@st.cache_data
def alle_daten_holen():
    return database.laden()

@st.cache_data
def stadtdaten_holen():
    return database.stadtdaten_laden()

@st.cache_data(ttl=3600)   # 1 Stunde cachen — SNB aktualisiert nicht öfter
def zinsen_holen():
    return data_fetcher.hypothekarzinsen_holen()

# ── Seite 1: Preisschätzung ───────────────────────────────────────────────────

def seite_preisschaetzung():
    st.header("Immobilienpreise schätzen")

    # ── Suche und Preistyp ────────────────────────────────────────────────────
    col_suche, col_typ = st.columns([2, 1])
    with col_suche:
        eingabe = st.text_input("🔍 PLZ oder Stadtname eingeben", placeholder="z. B. 8001 oder Zürich")
    with col_typ:
        typ_wahl = st.radio("Preistyp", ["Kaufpreis", "Mietpreis"], horizontal=True)

    # Stadtauswahl: Sucheingabe vorfiltern, dann Dropdown
    gefundene_stadt = suche_stadt(eingabe, STADTLISTE) if eingabe else None
    if eingabe and not gefundene_stadt:
        st.caption("Stadt nicht gefunden – bitte unten manuell auswählen.")

    standard_index = STADTLISTE.index(gefundene_stadt) if gefundene_stadt else 0
    stadt = st.selectbox("Stadt", STADTLISTE, index=standard_index)

    # ── Immobilien-Eigenschaften ──────────────────────────────────────────────
    st.markdown("#### Angaben zur Immobilie")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        zimmer = st.select_slider(
            "Anzahl Zimmer", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0], value=3.5
        )
        flaeche = st.slider("Wohnfläche (m²)", 25, 300, 85)
    with col_b:
        stockwerk = st.slider("Stockwerk", 0, 15, 2)
        baujahr = st.slider("Baujahr", 1900, 2024, 2000)
    with col_c:
        parkplatz = st.checkbox("Parkplatz vorhanden", value=False)
        st.write("")
        st.write("")
        berechnen = st.button("Preis schätzen", type="primary", use_container_width=True)

    # ── Ergebnis anzeigen ─────────────────────────────────────────────────────
    typ_intern = "kauf" if typ_wahl == "Kaufpreis" else "miete"
    preis = ml_model.schaetzen(stadt, flaeche, zimmer, stockwerk, parkplatz, baujahr, typ_intern)

    if berechnen:
        st.session_state["letzter_preis"] = preis

    st.markdown("---")
    st.markdown(f"### Geschätzter {typ_bezeichnung(typ_intern)}")

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.metric(
            label=f"{zimmer}-Zimmer-Wohnung, {flaeche} m² in {stadt}",
            value=chf(preis),
        )
        if typ_intern == "kauf":
            preis_m2 = preis / flaeche
            st.caption(f"ca. {chf(preis_m2)} pro m²")
        else:
            st.caption(f"ca. {chf(preis * 12)} pro Jahr")

    # Gauge: Preisvergleich zum Schweizer Durchschnitt
    daten = daten_holen(typ_intern)
    schweizer_schnitt = daten["preis"].mean()
    st.info(f"Schweizer Durchschnitt: {chf(schweizer_schnitt)}")
    
    # Wechselkurs anzeigen
    # ── Wechselkurs ──────────────────────────────────────────────────────────
    eur_kurs, usd_kurs = data_fetcher.wechselkurs_holen()
    st.markdown(f"**Preis in anderen Währungen:** € {preis * eur_kurs:,.0f} EUR | $ {preis * usd_kurs:,.0f} USD")

    # ── SNB Hypothekarzinsen ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Aktuelle SNB Hypothekarzinsen")
    zinsen = zinsen_holen()
    if zinsen is not None and not zinsen.empty:
        col_z1, col_z2 = st.columns([1, 2])
        with col_z1:
            # Tabelle mit aktuellen Zinssätzen
            anzeige = zinsen[["bezeichnung", "wert"]].copy()
            anzeige.columns = ["Produkt", "Zinssatz (%)"]
            anzeige["Zinssatz (%)"] = anzeige["Zinssatz (%)"].map(lambda x: f"{x:.2f}%")
            st.dataframe(anzeige, hide_index=True, use_container_width=True)
        with col_z2:
            # Balkendiagramm der Zinssätze
            balken_zinsen = px.bar(
                zinsen, x="bezeichnung", y="wert",
                color="wert", color_continuous_scale="Blues",
                labels={"bezeichnung": "Produkt", "wert": "Zinssatz (%)"},
                text=zinsen["wert"].map(lambda x: f"{x:.2f}%"),
            )
            balken_zinsen.update_layout(
                showlegend=False, coloraxis_showscale=False,
                margin={"t": 10, "b": 10}
            )
            balken_zinsen.update_traces(textposition="outside")
            st.plotly_chart(balken_zinsen, use_container_width=True)
        # Datum des letzten SNB-Updates anzeigen
        letztes_datum = zinsen["datum"].dropna()
        if not letztes_datum.empty:
            st.caption(f"Quelle: SNB Datenportal · Stand: {letztes_datum.max().strftime('%B %Y')}")
    
    

# ── Seite 2: Marktübersicht ───────────────────────────────────────────────────

def seite_markt():
    st.header("Marktübersicht Schweiz")

    typ_wahl = st.radio("Preistyp anzeigen:", ["Kaufpreise", "Mietpreise"], horizontal=True)
    typ_intern = "kauf" if typ_wahl == "Kaufpreise" else "miete"
    einheit = "CHF" if typ_intern == "kauf" else "CHF/Mt."

    daten = daten_holen(typ_intern)
    stadtdaten = stadtdaten_holen()

    # Stadtdurchschnitte für Karte und Balkendiagramm
    agg = (
        daten.groupby("stadt", as_index=False)
        .agg(durchschnitt=("preis", "mean"), anzahl=("id", "count"))
        .merge(stadtdaten[["stadt", "lat", "lon", "kanton"]], on="stadt", how="left")
    )

    # ── Karte der Schweiz ─────────────────────────────────────────────────────
    st.subheader("Preiskarte Schweiz")
    karte = px.scatter_mapbox(
        agg,
        lat="lat", lon="lon",
        size="durchschnitt",
        color="durchschnitt",
        hover_name="stadt",
        hover_data={"kanton": True, "durchschnitt": ":,.0f", "anzahl": True, "lat": False, "lon": False},
        color_continuous_scale="Greens",
        zoom=6.5, height=430, size_max=42,
        labels={"durchschnitt": f"Ø {einheit}", "anzahl": "Inserate"},
    )
    karte.update_layout(mapbox_style="open-street-map", margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(karte, use_container_width=True)

    col_links, col_rechts = st.columns(2)

    # ── Balkendiagramm: Städtevergleich ──────────────────────────────────────
    with col_links:
        st.subheader("Durchschnittspreise nach Stadt")
        balken = px.bar(
            agg.sort_values("durchschnitt", ascending=True),
            x="durchschnitt", y="stadt",
            orientation="h",
            color="durchschnitt",
            color_continuous_scale="Greens",
            text_auto=",.0f",
            labels={"durchschnitt": f"Ø Preis ({einheit})", "stadt": "Stadt"},
        )
        balken.update_layout(showlegend=False, coloraxis_showscale=False)
        balken.update_traces(textposition="outside")
        st.plotly_chart(balken, use_container_width=True)

    # ── Scatter Plot: Fläche vs. Preis mit Trendlinie ─────────────────────────
    with col_rechts:
        st.subheader("Wohnfläche vs. Preis")
        scatter = px.scatter(
            daten, x="flaeche", y="preis",
            color="stadt", opacity=0.55,
            labels={"flaeche": "Wohnfläche (m²)", "preis": f"Preis ({einheit})", "stadt": "Stadt"},
        )
        # Trendlinie manuell berechnen (ohne externe Abhängigkeit)
        if len(daten) > 2:
            x_werte = daten["flaeche"].values
            y_werte = daten["preis"].values
            koeffizienten = np.polyfit(x_werte, y_werte, 1)
            poly = np.poly1d(koeffizienten)
            x_linie = np.linspace(x_werte.min(), x_werte.max(), 100)
            scatter.add_trace(go.Scatter(
                x=x_linie, y=poly(x_linie),
                mode="lines", name="Trend",
                line={"color": "#1b5e20", "width": 2, "dash": "dash"},
            ))
        scatter.update_layout(legend={"font": {"size": 9}})
        st.plotly_chart(scatter, use_container_width=True)

# ── Seite 3: Meine Immobilien ─────────────────────────────────────────────────

def seite_immobilien():
    st.header("Meine Immobilien")

    # ── Formular: neue Immobilie hinzufügen ───────────────────────────────────
    st.subheader("Immobilie eintragen")
    with st.form("neue_immobilie"):
        col1, col2, col3 = st.columns(3)
        with col1:
            stadt = st.selectbox("Stadt", STADTLISTE)
            flaeche = st.number_input("Wohnfläche (m²)", min_value=20.0, max_value=500.0, value=80.0, step=5.0)
            zimmer = st.select_slider("Zimmer", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], value=3.0)
        with col2:
            stockwerk = st.number_input("Stockwerk", min_value=0, max_value=30, value=2)
            baujahr = st.number_input("Baujahr", min_value=1850, max_value=2024, value=2000)
            parkplatz = st.checkbox("Parkplatz vorhanden")
        with col3:
            typ_eingabe = st.radio("Preistyp", ["Kaufpreis", "Mietpreis"])
            if typ_eingabe == "Kaufpreis":
                preis = st.number_input("Preis (CHF)", min_value=100000, max_value=10000000, value=800000, step=10000)
            else:
                preis = st.number_input("Monatliche Miete (CHF)", min_value=500, max_value=20000, value=2000, step=50)
            speichern = st.form_submit_button("Eintrag speichern", use_container_width=True)

    if speichern:
        typ_intern = "kauf" if typ_eingabe == "Kaufpreis" else "miete"
        database.einfuegen(stadt, flaeche, zimmer, stockwerk, int(parkplatz), baujahr, preis, typ_intern)
        # Cache leeren damit neue Einträge sofort erscheinen
        daten_holen.clear()
        alle_daten_holen.clear()
        st.success("Immobilie wurde gespeichert.")

    # ── Liste: gespeicherte Immobilien ────────────────────────────────────────
    st.markdown("---")
    st.subheader("Gespeicherte Einträge")

    typ_filter = st.radio("Anzeigen:", ["Alle", "Nur Kauf", "Nur Miete"], horizontal=True)
    if typ_filter == "Nur Kauf":
        anzeige_daten = database.laden("kauf")
    elif typ_filter == "Nur Miete":
        anzeige_daten = database.laden("miete")
    else:
        anzeige_daten = database.laden()

    # Spalten auf Deutsch umbenennen für die Anzeige
    anzeige = anzeige_daten.rename(columns={
        "stadt": "Stadt", "kanton": "Kanton", "flaeche": "Fläche (m²)",
        "zimmer": "Zimmer", "stockwerk": "Stockwerk", "parkplatz": "Parkplatz",
        "baujahr": "Baujahr", "preis": "Preis (CHF)", "typ": "Typ",
    }).drop(columns=["id"], errors="ignore")
    anzeige.insert(0, "Nr.", range(1, len(anzeige) + 1))
    anzeige["Typ"] = anzeige["Typ"].map({"kauf": "Kauf", "miete": "Miete"})
    anzeige["Parkplatz"] = anzeige["Parkplatz"].map({1: "Ja", 0: "Nein"})

    st.dataframe(anzeige, use_container_width=True, hide_index=True)

    # Lösch-Buttons pro Zeile
    st.markdown("---")
    for nr, (_, zeile) in enumerate(anzeige_daten.iterrows(), start=1):
        col1, col2 = st.columns([8, 1])
        with col1:
            st.write(f"#{nr} | {zeile['stadt']} | {zeile['flaeche']} m² | {zeile['zimmer']} Zi. | {zeile['baujahr']} | CHF {zeile['preis']:,.0f}")
        with col2:
            if st.button("🗑️", key=f"loeschen_{zeile['id']}"):
                conn = database.verbindung()
                conn.execute("DELETE FROM immobilien WHERE id=?", (zeile['id'],))
                conn.commit()
                conn.close()
                st.cache_data.clear()
                st.rerun()
   

# ── Navigation ────────────────────────────────────────────────────────────────

st.sidebar.title(APP_NAME)
st.sidebar.caption(APP_UNTERTITEL)
st.sidebar.markdown("---")
seite = st.sidebar.radio("Navigation", SEITEN, label_visibility="collapsed")

if seite == SEITEN[0]:
    seite_preisschaetzung()
elif seite == SEITEN[1]:
    seite_markt()
else:
    seite_immobilien()
