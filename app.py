# app.py
# Tourism Experience Analytics
# Streamlit Application

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "master_tourism_data.csv"
)

VISIT_MODE_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "visit_mode_model.pkl"
)

RATING_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "rating_model.pkl"
)

USER_ITEM_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "user_item_matrix.pkl"
)

SIMILARITY_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "attraction_similarity.pkl"
)

ATTRACTION_INFO_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "attraction_info.pkl"
)


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():

    visit_mode_model = joblib.load(
        VISIT_MODE_MODEL_PATH
    )

    rating_model = joblib.load(
        RATING_MODEL_PATH
    )

    user_item_matrix = joblib.load(
        USER_ITEM_PATH
    )

    attraction_similarity = joblib.load(
        SIMILARITY_PATH
    )

    attraction_info = joblib.load(
        ATTRACTION_INFO_PATH
    )

    return (
        visit_mode_model,
        rating_model,
        user_item_matrix,
        attraction_similarity,
        attraction_info
    )


# Load everything
df = load_data()

(
    visit_mode_model,
    rating_model,
    user_item_matrix,
    attraction_similarity,
    attraction_info
) = load_models()


# ============================================================
# HELPER FUNCTION - RECOMMENDATIONS
# ============================================================

def recommend_attractions(user_id, top_n=5):

    if user_id not in user_item_matrix.index:
        return pd.DataFrame()

    user_ratings = user_item_matrix.loc[user_id]

    visited_attractions = user_ratings[
        user_ratings > 0
    ].index.tolist()

    liked_attractions = user_ratings[
        user_ratings >= 4
    ].index.tolist()

    if len(liked_attractions) == 0:
        liked_attractions = visited_attractions

    scores = {}

    for candidate in user_item_matrix.columns:

        if candidate in visited_attractions:
            continue

        score = 0.0
        similarity_sum = 0.0

        for liked in liked_attractions:

            similarity = attraction_similarity.loc[
                candidate,
                liked
            ]

            rating = user_ratings[liked]

            score += similarity * rating
            similarity_sum += abs(similarity)

        if similarity_sum > 0:

            scores[candidate] = (
                score / similarity_sum
            )

    if not scores:
        return pd.DataFrame()

    recommendations = (
        pd.Series(scores)
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    recommendations.columns = [
        "AttractionId",
        "RecommendationScore"
    ]

    recommendations = recommendations.merge(
        attraction_info.reset_index(),
        on="AttractionId",
        how="left"
    )

    return recommendations


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌍 Tourism Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Visit Mode Prediction",
        "Rating Prediction",
        "Personalized Recommendations"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("🌍 Tourism Experience Analytics")

    st.write(
        "An analytics and machine learning system for "
        "tourism experience analysis, prediction, and "
        "personalized attraction recommendations."
    )

    st.divider()

    # Key metrics
    total_visits = len(df)

    unique_users = df["UserId"].nunique()

    unique_attractions = df["AttractionId"].nunique()

    average_rating = df["Rating"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Visits",
        f"{total_visits:,}"
    )

    col2.metric(
        "Unique Users",
        f"{unique_users:,}"
    )

    col3.metric(
        "Attractions",
        unique_attractions
    )

    col4.metric(
        "Average Rating",
        f"{average_rating:.2f}/5"
    )

    st.divider()

    # Popular attractions
    st.subheader("⭐ Most Popular Attractions")

    popular_attractions = (
        df.groupby("Attraction")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="Visits")
    )

    st.dataframe(
        popular_attractions,
        use_container_width=True,
        hide_index=True
    )

    # Visit mode distribution
    st.subheader("👥 Visit Mode Distribution")

    visit_mode_counts = (
        df["VisitMode"]
        .value_counts()
        .reset_index()
    )

    visit_mode_counts.columns = [
        "VisitMode",
        "Visits"
    ]

    st.bar_chart(
        visit_mode_counts.set_index("VisitMode")
    )

    # Top regions
    st.subheader("🌎 Top Tourism Regions")

    top_regions = (
        df.groupby("Region")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="Visits")
    )

    st.dataframe(
        top_regions,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# VISIT MODE PREDICTION
# ============================================================

elif page == "Visit Mode Prediction":

    st.title("🎯 Visit Mode Prediction")

    st.write(
        "Predict the tourist's visit mode using "
        "visit, attraction, and geographic information."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        visit_year = st.number_input(
            "Visit Year",
            min_value=2013,
            max_value=2030,
            value=2022
        )

        visit_month = st.selectbox(
            "Visit Month",
            list(range(1, 13)),
            index=6
        )

        attraction_name = st.selectbox(
            "Select Attraction",
            sorted(
                df["Attraction"]
                .dropna()
                .unique()
            )
        )

    # Get attraction details
    attraction_row = df[
        df["Attraction"] == attraction_name
    ].iloc[0]

    with col2:

        st.text_input(
            "Attraction Type",
            value=str(
                attraction_row["AttractionType"]
            ),
            disabled=True
        )

        st.text_input(
            "Attraction City",
            value=str(
                attraction_row["AttractionCity"]
            ),
            disabled=True
        )

        st.text_input(
            "Country",
            value=str(
                attraction_row["Country"]
            ),
            disabled=True
        )

    if st.button(
        "Predict Visit Mode",
        type="primary"
    ):

        input_data = pd.DataFrame(
            {
                "VisitYear": [visit_year],
                "VisitMonth": [visit_month],
                "AttractionId": [
                    attraction_row["AttractionId"]
                ],
                "AttractionTypeId": [
                    attraction_row["AttractionTypeId"]
                ],
                "AttractionCityId": [
                    attraction_row["AttractionCityId"]
                ],
                "ContinentId": [
                    attraction_row["ContinentId"]
                ],
                "RegionId": [
                    attraction_row["RegionId"]
                ],
                "CountryId": [
                    attraction_row["CountryId"]
                ],
                "CityId": [
                    attraction_row["CityId"]
                    if pd.notna(
                        attraction_row["CityId"]
                    )
                    else -1
                ]
            }
        )

        prediction = visit_mode_model.predict(
            input_data
        )[0]

        st.success(
            f"Predicted Visit Mode: **{prediction}**"
        )


# ============================================================
# RATING PREDICTION
# ============================================================

elif page == "Rating Prediction":

    st.title("⭐ Attraction Rating Prediction")

    st.write(
        "Predict the expected tourist rating "
        "for a selected attraction."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        visit_year = st.number_input(
            "Visit Year",
            min_value=2013,
            max_value=2030,
            value=2022,
            key="rating_year"
        )

        visit_month = st.selectbox(
            "Visit Month",
            list(range(1, 13)),
            index=6,
            key="rating_month"
        )

        visit_mode = st.selectbox(
            "Visit Mode",
            sorted(
                df["VisitMode"]
                .dropna()
                .unique()
            )
        )

    with col2:

        attraction_name = st.selectbox(
            "Select Attraction",
            sorted(
                df["Attraction"]
                .dropna()
                .unique()
            ),
            key="rating_attraction"
        )

    attraction_row = df[
        df["Attraction"] == attraction_name
    ].iloc[0]

    st.write(
        f"**Attraction Type:** "
        f"{attraction_row['AttractionType']}"
    )

    st.write(
        f"**Location:** "
        f"{attraction_row['AttractionCity']}, "
        f"{attraction_row['Country']}"
    )

    if st.button(
        "Predict Rating",
        type="primary"
    ):

        input_data = pd.DataFrame(
            {
                "VisitYear": [visit_year],
                "VisitMonth": [visit_month],
                "VisitMode": [visit_mode],
                "AttractionId": [
                    attraction_row["AttractionId"]
                ],
                "AttractionTypeId": [
                    attraction_row["AttractionTypeId"]
                ],
                "AttractionCityId": [
                    attraction_row["AttractionCityId"]
                ],
                "ContinentId": [
                    attraction_row["ContinentId"]
                ],
                "RegionId": [
                    attraction_row["RegionId"]
                ],
                "CountryId": [
                    attraction_row["CountryId"]
                ],
                "CityId": [
                    attraction_row["CityId"]
                    if pd.notna(
                        attraction_row["CityId"]
                    )
                    else -1
                ]
            }
        )

        predicted_rating = rating_model.predict(
            input_data
        )[0]

        # Keep prediction within rating scale
        predicted_rating = np.clip(
            predicted_rating,
            1,
            5
        )

        st.success(
            f"Predicted Rating: "
            f"**{predicted_rating:.2f} / 5**"
        )


# ============================================================
# PERSONALIZED RECOMMENDATIONS
# ============================================================

elif page == "Personalized Recommendations":

    st.title("✨ Personalized Attraction Recommendations")

    st.write(
        "Get personalized attraction recommendations "
        "based on the user's historical ratings."
    )

    st.divider()

    user_ids = user_item_matrix.index.tolist()

    selected_user = st.selectbox(
        "Select User ID",
        user_ids
    )

    if st.button(
        "Generate Recommendations",
        type="primary"
    ):

        recommendations = recommend_attractions(
            selected_user,
            top_n=5
        )

        if recommendations.empty:

            st.warning(
                "No personalized recommendations "
                "are available for this user."
            )

        else:

            st.subheader(
                f"Top 5 Recommendations for User "
                f"{selected_user}"
            )

            display_df = recommendations.copy()

            display_df["RecommendationScore"] = (
                display_df["RecommendationScore"]
                .round(3)
            )

            display_df = display_df[
                [
                    "Attraction",
                    "AttractionType",
                    "AttractionCity",
                    "Country",
                    "Region",
                    "RecommendationScore"
                ]
            ]

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Tourism Experience Analytics | "
    "Classification • Regression • Recommendation"
)