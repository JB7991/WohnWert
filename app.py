"""Streamlit entry point for ImmoInsight - Swiss Real Estate Price Estimator."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import database
import ml_model
from config import (
    APP_SUBTITLE,
    APP_TITLE,
    CITY_LIST,
    CITY_PROFILES,
    CURRENT_YEAR,
    PAGE_OPTIONS,
    PLOTLY_GREEN_SCALE,
    TEAM_CONTRIBUTIONS,
)
from data_fetcher import enrich_city_stats
from utils import (
    calculate_summary_statistics,
    coerce_bool,
    current_timestamp,
    dataframe_to_csv_bytes,
    format_chf,
    metric_card,
    price_position_label,
)


def configure_page() -> None:
    """Configure Streamlit page metadata and inject project-specific CSS.

    Parameters:
        None.

    Returns:
        None.
    """
    st.set_page_config(page_title=APP_TITLE, page_icon="🏡", layout="wide")

    # The CSS keeps the dashboard understated while giving key metrics a card-like treatment.
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        .metric-card {
            border: 1px solid rgba(28, 107, 64, 0.16);
            border-radius: 8px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.055);
            padding: 1rem 1.1rem;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbf8 100%);
            min-height: 110px;
        }
        .metric-card .metric-label {
            color: #3f5f4b;
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .metric-card .metric-value {
            color: #153d26;
            font-size: 1.75rem;
            font-weight: 700;
            line-height: 1.15;
        }
        .metric-card .metric-subtitle {
            color: #587262;
            font-size: 0.86rem;
            margin-top: 0.45rem;
        }
        div[data-testid="stToast"] { border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def initialize_app_database() -> None:
    """Create database tables and seed first-run data once per Streamlit process.

    Parameters:
        None.

    Returns:
        None.
    """
    database.initialize_database()


def get_data_signature() -> str:
    """Return a compact database signature used to invalidate Streamlit caches.

    Parameters:
        None.

    Returns:
        A string containing row-count and timestamp information.
    """
    return database.get_data_signature()


@st.cache_data(show_spinner=False)
def load_properties_cached(signature: str) -> pd.DataFrame:
    """Load property rows from SQLite with Streamlit data caching.

    Parameters:
        signature: Database signature that invalidates the cached dataframe.

    Returns:
        A dataframe containing property records.
    """
    _ = signature
    return database.get_properties()


@st.cache_data(show_spinner=False)
def load_training_data_cached(signature: str) -> pd.DataFrame:
    """Load model training data from SQLite with Streamlit data caching.

    Parameters:
        signature: Database signature that invalidates the cached dataframe.

    Returns:
        A dataframe with properties joined to city statistics.
    """
    _ = signature
    return database.get_training_data()


@st.cache_data(show_spinner=False)
def load_city_stats_cached(signature: str) -> pd.DataFrame:
    """Load city statistics from SQLite with Streamlit data caching.

    Parameters:
        signature: Database signature that invalidates the cached dataframe.

    Returns:
        A dataframe containing city-level enrichment data.
    """
    _ = signature
    return database.get_city_stats()


@st.cache_data(show_spinner=False)
def load_model_runs_cached(signature: str) -> pd.DataFrame:
    """Load model training history from SQLite with Streamlit data caching.

    Parameters:
        signature: Database signature that invalidates the cached dataframe.

    Returns:
        A dataframe containing persisted model metrics.
    """
    _ = signature
    return database.get_model_runs()


@st.cache_resource(show_spinner=False)
def load_model_bundle_cached(signature: str) -> dict[str, Any]:
    """Load the cached ML bundle or train it once when no joblib file exists.

    Parameters:
        signature: Database signature that controls Streamlit resource caching.

    Returns:
        A dictionary containing fitted pipelines, metrics, and split metadata.
    """
    training_df = load_training_data_cached(signature)
    return ml_model.load_or_train_model_bundle(training_df)


def clear_data_caches() -> None:
    """Clear Streamlit data caches after a database write.

    Parameters:
        None.

    Returns:
        None.
    """
    load_properties_cached.clear()
    load_training_data_cached.clear()
    load_city_stats_cached.clear()
    load_model_runs_cached.clear()


def render_header() -> None:
    """Render the global application header.

    Parameters:
        None.

    Returns:
        None.
    """
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)


def render_sidebar_api_refresh() -> None:
    """Render a sidebar control for public API enrichment of city statistics.

    Parameters:
        None.

    Returns:
        None.
    """
    with st.sidebar.expander("Public data refresh", expanded=False):
        st.write("Refresh geocoding and demographic density from Nominatim and Wikidata.")
        if st.button("Refresh city stats from APIs", use_container_width=True):
            with st.spinner("Calling public APIs and updating city statistics..."):
                enriched_stats, warnings = enrich_city_stats(CITY_LIST)
                updated_rows = database.upsert_city_stats(enriched_stats)
                clear_data_caches()

            # API failures do not break the app because deterministic fallback data remains available.
            if warnings:
                for warning in warnings[:4]:
                    st.warning(warning)
                if len(warnings) > 4:
                    st.warning(f"{len(warnings) - 4} additional API warnings were suppressed.")
            st.toast(f"Updated {updated_rows} city-stat rows.", icon="✅")


def get_selected_model(bundle: dict[str, Any], key: str = "model_selector") -> str:
    """Render a model selector and return the selected model name.

    Parameters:
        bundle: The fitted model bundle returned by ``ml_model``.
        key: Streamlit widget key.

    Returns:
        The selected model name.
    """
    model_names = list(bundle["models"].keys())
    default_index = model_names.index("Random Forest") if "Random Forest" in model_names else 0
    return st.selectbox("Prediction model", model_names, index=default_index, key=key)


def build_property_feature_row(
    city: str,
    area_m2: float,
    rooms: float,
    floor: int,
    has_parking: bool,
    has_garden: bool,
    year_built: int,
    city_stats: pd.DataFrame,
) -> dict[str, Any]:
    """Build a model-ready feature row from user inputs and city statistics.

    Parameters:
        city: Swiss city name selected by the user.
        area_m2: Living area in square meters.
        rooms: Number of rooms.
        floor: Floor number.
        has_parking: Whether the property includes parking.
        has_garden: Whether the property includes a garden.
        year_built: Construction year.
        city_stats: City statistics dataframe from SQLite.

    Returns:
        A dictionary matching the ML pipeline feature schema.
    """
    stats_row = city_stats.loc[city_stats["city"] == city].iloc[0]

    # The ML layer expects numeric boolean indicators and the joined city-level statistics.
    return {
        "city": city,
        "area_m2": float(area_m2),
        "rooms": float(rooms),
        "floor": int(floor),
        "has_parking": int(has_parking),
        "has_garden": int(has_garden),
        "year_built": int(year_built),
        "population_density": float(stats_row["population_density"]),
        "avg_income": float(stats_row["avg_income"]),
    }


def render_price_estimator(bundle: dict[str, Any], signature: str) -> None:
    """Render the price-estimation workflow.

    Parameters:
        bundle: Fitted model bundle for prediction.
        signature: Database signature used to load cached data.

    Returns:
        None.
    """
    st.subheader("🏠 Price Estimator")
    city_stats = load_city_stats_cached(signature)
    properties = load_properties_cached(signature)

    # The form groups all property features in one predictable input surface.
    selected_model = get_selected_model(bundle, key="price_model_selector")
    with st.form("price_estimator_form"):
        col_left, col_mid, col_right = st.columns(3)
        with col_left:
            city = st.selectbox("City", CITY_LIST, index=0)
            area_m2 = st.slider("Area (m²)", 25.0, 320.0, 92.0, 1.0)
            rooms = st.slider("Rooms", 1.0, 8.0, 3.5, 0.5)
        with col_mid:
            floor = st.slider("Floor", 0, 35, 3, 1)
            year_built = st.slider("Year Built", 1900, CURRENT_YEAR, 1998, 1)
            has_parking = st.checkbox("Has Parking", value=True)
        with col_right:
            has_garden = st.checkbox("Has Garden", value=False)
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Estimate Price", use_container_width=True)

    # A first prediction is shown immediately; submitting simply refreshes session state explicitly.
    feature_row = build_property_feature_row(
        city=city,
        area_m2=area_m2,
        rooms=rooms,
        floor=floor,
        has_parking=has_parking,
        has_garden=has_garden,
        year_built=year_built,
        city_stats=city_stats,
    )
    prediction = ml_model.predict_price(bundle, selected_model, feature_row)
    if submitted:
        st.session_state["last_prediction"] = prediction
        st.toast("Price estimate refreshed.", icon="🏡")

    predicted_price = float(prediction["prediction"])
    lower_bound = float(prediction["lower_bound"])
    upper_bound = float(prediction["upper_bound"])
    price_per_m2 = predicted_price / float(area_m2)

    # City-average context comes from the local SQLite market sample.
    city_prices = properties.loc[properties["city"] == city, "estimated_price"]
    city_average = float(city_prices.mean()) if not city_prices.empty else predicted_price
    city_std = float(city_prices.std(ddof=0)) if len(city_prices) > 1 else predicted_price * 0.12
    position_label = price_position_label(predicted_price, city_average, city_std)

    metric_cols = st.columns(3)
    metric_cols[0].markdown(
        metric_card("Predicted price", format_chf(predicted_price), f"{selected_model} estimate"),
        unsafe_allow_html=True,
    )
    metric_cols[1].markdown(
        metric_card("Price per m²", format_chf(price_per_m2), "Model-implied unit price"),
        unsafe_allow_html=True,
    )
    metric_cols[2].markdown(
        metric_card(
            "Confidence interval",
            f"{format_chf(lower_bound)} - {format_chf(upper_bound)}",
            "Approximate 80% prediction interval",
        ),
        unsafe_allow_html=True,
    )

    # The gauge makes the city-average comparison legible at a glance.
    gauge_max = max(predicted_price, city_average) * 1.55
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=predicted_price,
            number={"prefix": "CHF ", "valueformat": ",.0f"},
            delta={"reference": city_average, "relative": True, "valueformat": ".1%"},
            title={"text": f"{city}: {position_label} city average"},
            gauge={
                "axis": {"range": [0, gauge_max], "tickformat": ",.0f"},
                "bar": {"color": "#238b45"},
                "steps": [
                    {"range": [0, city_average * 0.9], "color": "#e5f5e0"},
                    {"range": [city_average * 0.9, city_average * 1.1], "color": "#a1d99b"},
                    {"range": [city_average * 1.1, gauge_max], "color": "#41ab5d"},
                ],
                "threshold": {"line": {"color": "#145a32", "width": 4}, "value": city_average},
            },
        )
    )
    gauge_fig.update_layout(height=360, margin={"l": 20, "r": 20, "t": 60, "b": 10})
    st.plotly_chart(gauge_fig, use_container_width=True)

    if st.button("Save this estimate", type="primary", use_container_width=True):
        stats_row = city_stats.loc[city_stats["city"] == city].iloc[0]
        database.insert_property(
            city=city,
            canton=str(stats_row["canton"]),
            area_m2=float(area_m2),
            rooms=float(rooms),
            floor=int(floor),
            has_parking=coerce_bool(has_parking),
            has_garden=coerce_bool(has_garden),
            year_built=int(year_built),
            estimated_price=predicted_price,
            timestamp=current_timestamp(),
        )
        clear_data_caches()
        st.toast("Estimate saved to SQLite.", icon="✅")


def render_market_overview(signature: str) -> None:
    """Render market-level visual analytics.

    Parameters:
        signature: Database signature used to load cached data.

    Returns:
        None.
    """
    st.subheader("📊 Market Overview")
    properties = load_properties_cached(signature)
    city_stats = load_city_stats_cached(signature)

    selected_cities = st.sidebar.multiselect(
        "Market cities",
        CITY_LIST,
        default=CITY_LIST,
        help="All charts on this page update from this city filter.",
    )
    filtered = properties.loc[properties["city"].isin(selected_cities)].copy()

    if filtered.empty:
        st.warning("No properties match the selected cities.")
        return

    # Aggregations power the map bubbles and bar-chart error bars.
    city_agg = (
        filtered.groupby("city", as_index=False)
        .agg(avg_price=("estimated_price", "mean"), std_price=("estimated_price", "std"), listings=("id", "count"))
        .merge(city_stats[["city", "lat", "lon", "canton"]], on="city", how="left")
    )
    city_agg["std_price"] = city_agg["std_price"].fillna(0.0)

    map_fig = px.scatter_mapbox(
        city_agg,
        lat="lat",
        lon="lon",
        size="avg_price",
        color="avg_price",
        hover_name="city",
        hover_data={"canton": True, "avg_price": ":,.0f", "std_price": ":,.0f", "listings": True},
        color_continuous_scale=PLOTLY_GREEN_SCALE,
        zoom=6,
        height=450,
        size_max=42,
    )
    map_fig.update_layout(mapbox_style="open-street-map", margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(map_fig, use_container_width=True)

    col_bar, col_scatter = st.columns(2)
    with col_bar:
        bar_fig = px.bar(
            city_agg.sort_values("avg_price", ascending=False),
            x="city",
            y="avg_price",
            error_y="std_price",
            color="avg_price",
            color_continuous_scale=PLOTLY_GREEN_SCALE,
            labels={"avg_price": "Average price (CHF)", "city": "City"},
            title="Average price per city",
        )
        bar_fig.update_layout(showlegend=False)
        st.plotly_chart(bar_fig, use_container_width=True)

    with col_scatter:
        scatter_fig = px.scatter(
            filtered,
            x="area_m2",
            y="estimated_price",
            color="city",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"area_m2": "Area (m²)", "estimated_price": "Price (CHF)"},
            title="Area vs. price with regression line",
            opacity=0.72,
        )
        if len(filtered) > 1 and filtered["area_m2"].nunique() > 1:
            slope, intercept = np.polyfit(filtered["area_m2"], filtered["estimated_price"], deg=1)
            x_line = np.array([filtered["area_m2"].min(), filtered["area_m2"].max()])
            y_line = slope * x_line + intercept
            scatter_fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Regression line",
                    line={"color": "#145a32", "width": 3},
                )
            )
        st.plotly_chart(scatter_fig, use_container_width=True)

    # Rooms are bucketed as labels so half-room apartments remain visible in the heatmap.
    heatmap_data = filtered.copy()
    heatmap_data["rooms_bucket"] = heatmap_data["rooms"].map(lambda value: f"{value:g}")
    heatmap_pivot = heatmap_data.pivot_table(
        index="city",
        columns="rooms_bucket",
        values="estimated_price",
        aggfunc="mean",
    )
    heatmap_fig = px.imshow(
        heatmap_pivot,
        aspect="auto",
        color_continuous_scale=PLOTLY_GREEN_SCALE,
        labels={"color": "Avg. price (CHF)", "x": "Rooms", "y": "City"},
        title="Price heatmap by city and rooms",
        text_auto=",.0f",
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)


def render_data_explorer(signature: str) -> None:
    """Render the SQLite-backed data exploration and manual-entry page.

    Parameters:
        signature: Database signature used to load cached data.

    Returns:
        None.
    """
    st.subheader("🔍 Data Explorer")
    properties = load_properties_cached(signature)

    # Sidebar controls constrain both the table and the downloadable export.
    selected_cities = st.sidebar.multiselect("Explorer cities", CITY_LIST, default=CITY_LIST)
    min_price = int(properties["estimated_price"].min())
    max_price = int(properties["estimated_price"].max())
    min_area = float(properties["area_m2"].min())
    max_area = float(properties["area_m2"].max())
    price_range = st.sidebar.slider("Price range (CHF)", min_price, max_price, (min_price, max_price), step=10_000)
    area_range = st.sidebar.slider("Area range (m²)", min_area, max_area, (min_area, max_area), step=1.0)

    filtered = database.query_filtered_properties(
        cities=selected_cities,
        min_price=float(price_range[0]),
        max_price=float(price_range[1]),
        min_area=float(area_range[0]),
        max_area=float(area_range[1]),
    )

    summary = calculate_summary_statistics(filtered)
    summary_cols = st.columns(3)
    for column, metric_name in zip(summary_cols, ["estimated_price", "price_per_m2", "area_m2"], strict=False):
        row = summary.loc[summary["metric"] == metric_name]
        if not row.empty:
            column.markdown(
                metric_card(
                    metric_name.replace("_", " ").title(),
                    format_chf(float(row.iloc[0]["mean"])) if metric_name != "area_m2" else f"{row.iloc[0]['mean']:.1f} m²",
                    f"Median {row.iloc[0]['median']:,.1f} | Std {row.iloc[0]['std']:,.1f}",
                ),
                unsafe_allow_html=True,
            )

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Export filtered data as CSV",
        data=dataframe_to_csv_bytes(filtered),
        file_name=f"immoinsight_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    st.markdown("#### Add a property manually")
    city_stats = load_city_stats_cached(signature)
    with st.form("manual_property_entry"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            city = st.selectbox("City", CITY_LIST, key="manual_city")
            area_m2 = st.number_input("Area (m²)", min_value=20.0, max_value=500.0, value=85.0, step=1.0)
            rooms = st.number_input("Rooms", min_value=1.0, max_value=12.0, value=3.5, step=0.5)
        with col_b:
            floor = st.number_input("Floor", min_value=0, max_value=50, value=2, step=1)
            year_built = st.number_input("Year Built", min_value=1850, max_value=CURRENT_YEAR, value=2005, step=1)
            estimated_price = st.number_input("Known price / estimate (CHF)", min_value=100_000, max_value=10_000_000, value=850_000, step=10_000)
        with col_c:
            has_parking = st.checkbox("Has Parking", value=True, key="manual_parking")
            has_garden = st.checkbox("Has Garden", value=False, key="manual_garden")
            submitted = st.form_submit_button("Add property", use_container_width=True)

    if submitted:
        stats_row = city_stats.loc[city_stats["city"] == city].iloc[0]
        database.insert_property(
            city=city,
            canton=str(stats_row["canton"]),
            area_m2=float(area_m2),
            rooms=float(rooms),
            floor=int(floor),
            has_parking=coerce_bool(has_parking),
            has_garden=coerce_bool(has_garden),
            year_built=int(year_built),
            estimated_price=float(estimated_price),
            timestamp=current_timestamp(),
        )
        clear_data_caches()
        st.toast("Manual property saved.", icon="✅")


def render_model_training(bundle: dict[str, Any], signature: str) -> None:
    """Render model retraining, metrics, importances, and history.

    Parameters:
        bundle: Current fitted model bundle.
        signature: Database signature used to load cached data.

    Returns:
        None.
    """
    st.subheader("🤖 Model Training & Performance")

    if st.button("Retrain Model", type="primary", use_container_width=True):
        with st.spinner("Training Random Forest, Gradient Boosting, and Linear Regression models..."):
            training_df = database.get_training_data()
            bundle = ml_model.train_and_cache_models(training_df, save_runs=True)
            load_model_bundle_cached.clear()
            load_model_runs_cached.clear()
        st.toast("Models retrained and cached with joblib.", icon="✅")

    metrics_df = ml_model.metrics_to_dataframe(bundle)
    metrics_cols = st.columns(len(metrics_df))
    for column, (_, row) in zip(metrics_cols, metrics_df.iterrows(), strict=False):
        column.markdown(
            metric_card(
                str(row["model_type"]),
                f"R² {row['r2']:.3f}",
                f"MAE {format_chf(row['mae'])} | RMSE {format_chf(row['rmse'])}",
            ),
            unsafe_allow_html=True,
        )

    st.dataframe(
        metrics_df[["model_type", "mae", "rmse", "r2", "trained_at"]],
        use_container_width=True,
        hide_index=True,
    )

    split_col_a, split_col_b = st.columns(2)
    split_col_a.info(f"Training split size: {bundle['train_size']} rows")
    split_col_b.info(f"Test split size: {bundle['test_size']} rows")

    selected_model = get_selected_model(bundle, key="performance_model_selector")
    importance_df = ml_model.get_feature_importance(bundle, selected_model)
    importance_fig = px.bar(
        importance_df.sort_values("importance", ascending=True).tail(18),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=PLOTLY_GREEN_SCALE,
        labels={"importance": "Importance", "feature": "Feature"},
        title=f"Feature importance - {selected_model}",
    )
    importance_fig.update_layout(showlegend=False)
    st.plotly_chart(importance_fig, use_container_width=True)

    runs = load_model_runs_cached(signature)
    if runs.empty:
        st.info("No training history has been recorded yet.")
    else:
        history_fig = px.line(
            runs,
            x="trained_at",
            y="r2",
            color="model_type",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"trained_at": "Training time", "r2": "R²"},
            title="Training history: R² over time",
        )
        st.plotly_chart(history_fig, use_container_width=True)


def render_about_page() -> None:
    """Render project description, contribution matrix, and source documentation.

    Parameters:
        None.

    Returns:
        None.
    """
    st.subheader("📋 About & Contribution Matrix")

    default_markdown = """### Problem statement
Swiss home buyers and analysts need transparent first-pass price estimates that combine property features, city context, and observable market behavior. ImmoInsight demonstrates a reproducible data-science workflow: local persistence, public API enrichment, model comparison, and interactive decision support.

### Application scope
The stored property listings are synthetic seed data designed for teaching and model demonstration. Public APIs enrich city coordinates and demographic density where available, while the core estimator remains fully functional offline.
"""
    edited_markdown = st.text_area("Editable project markdown", value=default_markdown, height=220)
    st.markdown(edited_markdown)

    st.markdown("#### Contribution Matrix")
    contribution_df = pd.DataFrame(TEAM_CONTRIBUTIONS)
    st.dataframe(contribution_df, use_container_width=True, hide_index=True)

    st.markdown("#### Data sources and references")
    st.markdown(
        """
- **OpenStreetMap Nominatim API**: geocodes Swiss city names into latitude and longitude.
- **Wikidata Query Service**: retrieves public city-level population and area metadata used to refresh population-density values.
- **Swiss Federal Statistical Office / opendata.swiss**: documented as the authoritative Swiss open-data reference for demographic and economic context.
- **Synthetic seed listings**: generated locally on first startup with realistic Swiss city price ranges, documented in `database.py`.
"""
    )

    st.markdown("#### API usage documentation")
    st.markdown(
        """
The app does not require API keys. Public API calls are isolated in `data_fetcher.py`, use short timeouts, include a descriptive User-Agent, and never block core functionality. If an API is unavailable, the app keeps using the SQLite-backed fallback city statistics seeded from `config.py`.
"""
    )


def main() -> None:
    """Run the ImmoInsight Streamlit application.

    Parameters:
        None.

    Returns:
        None.
    """
    configure_page()
    with st.spinner("Preparing SQLite database and first-run seed data..."):
        initialize_app_database()

    signature = get_data_signature()
    bundle = load_model_bundle_cached(signature)

    render_header()
    st.sidebar.title("Navigation")
    render_sidebar_api_refresh()
    page = st.sidebar.radio("Page", PAGE_OPTIONS, index=0)

    # The sidebar radio keeps navigation lightweight while each page owns its page-specific filters.
    if page == "🏠 Price Estimator":
        render_price_estimator(bundle, signature)
    elif page == "📊 Market Overview":
        render_market_overview(signature)
    elif page == "🔍 Data Explorer":
        render_data_explorer(signature)
    elif page == "🤖 Model Training & Performance":
        render_model_training(bundle, signature)
    else:
        render_about_page()


if __name__ == "__main__":
    main()
