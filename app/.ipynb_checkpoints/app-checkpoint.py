import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib
from collections import defaultdict, deque


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UEFA Champions League Predictor",
    page_icon="⚽",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD SAVED ML MODEL
# ============================================================

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "champions_league_logistic_model.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "champions_league_scaler.pkl"
    )
)

feature_columns = joblib.load(
    os.path.join(
        MODEL_DIR,
        "champions_league_feature_columns.pkl"
    )
)


# ============================================================
# STANDARDIZE TEAM NAMES
# ============================================================

def standardize_team_name(team):

    mapping = {

        "Manchester City":
            "Manchester City FC (ENG)",

        "Barcelona":
            "FC Barcelona (ESP)",

        "Porto":
            "FC Porto (POR)",

        "Feyenoord":
            "Feyenoord Rotterdam (NED)"
    }

    return mapping.get(
        team.strip(),
        team.strip()
    )


# ============================================================
# PARSE MATCH FILE
# Supports:
# Team v Team 2-1
# Team 2-1 Team
# ============================================================

def parse_champions_league_file(
    filepath,
    season_name
):

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    parsed_matches = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # ----------------------------------------------------
        # FORMAT 1:
        # Team A v Team B 2-1
        # ----------------------------------------------------

        if " v " in line:

            line = re.sub(
                r"^\d{1,2}:\d{2}\s+",
                "",
                line
            )

            match = re.match(
                r"(.+?)\s+v\s+(.+?)\s+(\d+)-(\d+)",
                line
            )

            if not match:
                continue

            home_team = match.group(1).strip()
            away_team = match.group(2).strip()

            home_goals = int(
                match.group(3)
            )

            away_goals = int(
                match.group(4)
            )

        # ----------------------------------------------------
        # FORMAT 2:
        # Team A 2-1 Team B
        # ----------------------------------------------------

        else:

            match = re.match(
                r"^(.+?)\s+(\d+)-(\d+)\s+(.+?)$",
                line
            )

            if not match:
                continue

            home_team = match.group(1).strip()

            home_goals = int(
                match.group(2)
            )

            away_goals = int(
                match.group(3)
            )

            away_team = match.group(4).strip()

        # ----------------------------------------------------
        # Remove unwanted time/date text
        # ----------------------------------------------------

        home_team = re.sub(
            r"^\d{1,2}:\d{2}\s+",
            "",
            home_team
        ).strip()

        # ----------------------------------------------------
        # Determine result
        # ----------------------------------------------------

        if home_goals > away_goals:

            result = "Home Win"

        elif home_goals < away_goals:

            result = "Away Win"

        else:

            result = "Draw"

        parsed_matches.append({

            "home_team":
                home_team,

            "away_team":
                away_team,

            "home_goals":
                home_goals,

            "away_goals":
                away_goals,

            "result":
                result,

            "season":
                season_name
        })

    return pd.DataFrame(
        parsed_matches
    )


# ============================================================
# LOAD MATCH DATA
# ============================================================

@st.cache_data
def load_matches():

    season_files = [

        (
            "champions_league_2023_24.txt",
            "2023/24"
        ),

        (
            "champions_league_2024_25.txt",
            "2024/25"
        ),

        (
            "champions_league_2025_26.txt",
            "2025/26"
        ),

        (
            "champions_league_2026_27.txt",
            "2026/27"
        )
    ]

    all_matches = []

    for filename, season in season_files:

        filepath = os.path.join(
            DATA_DIR,
            filename
        )

        if not os.path.exists(filepath):
            continue

        df = parse_champions_league_file(
            filepath,
            season
        )

        if not df.empty:

            all_matches.append(
                df
            )

    if not all_matches:

        return pd.DataFrame()

    matches = pd.concat(
        all_matches,
        ignore_index=True
    )

    matches["home_team"] = (
        matches["home_team"]
        .apply(
            standardize_team_name
        )
    )

    matches["away_team"] = (
        matches["away_team"]
        .apply(
            standardize_team_name
        )
    )

    return matches


matches = load_matches()


# ============================================================
# STOP IF NO DATA
# ============================================================

if matches.empty:

    st.error(
        "No Champions League match data was found."
    )

    st.stop()


# ============================================================
# CURRENT FEATURE ENGINEERING
# ============================================================

def calculate_current_features(
    matches
):

    # --------------------------------------------------------
    # Separate histories
    # --------------------------------------------------------

    overall_history = defaultdict(
        lambda: deque(
            maxlen=5
        )
    )

    home_history = defaultdict(
        lambda: deque(
            maxlen=5
        )
    )

    away_history = defaultdict(
        lambda: deque(
            maxlen=5
        )
    )

    # --------------------------------------------------------
    # Elo ratings
    # --------------------------------------------------------

    elo_ratings = defaultdict(
        lambda: 1500.0
    )

    K_FACTOR = 20
    HOME_ADVANTAGE = 60

    # --------------------------------------------------------
    # Process matches chronologically
    # --------------------------------------------------------

    for _, match in matches.iterrows():

        home = match["home_team"]
        away = match["away_team"]

        home_goals = int(
            match["home_goals"]
        )

        away_goals = int(
            match["away_goals"]
        )

        # ====================================================
        # ELO BEFORE MATCH
        # ====================================================

        home_elo = elo_ratings[
            home
        ]

        away_elo = elo_ratings[
            away
        ]

        expected_home = 1 / (
            1 + 10 ** (
                (
                    away_elo
                    -
                    (
                        home_elo
                        +
                        HOME_ADVANTAGE
                    )
                )
                / 400
            )
        )

        # ====================================================
        # RESULT
        # ====================================================

        if home_goals > away_goals:

            actual_home = 1.0

            home_points = 3
            away_points = 0

            home_win = 1
            away_win = 0

        elif home_goals < away_goals:

            actual_home = 0.0

            home_points = 0
            away_points = 3

            home_win = 0
            away_win = 1

        else:

            actual_home = 0.5

            home_points = 1
            away_points = 1

            home_win = 0
            away_win = 0

        # ====================================================
        # UPDATE ELO AFTER MATCH
        # ====================================================

        elo_change = K_FACTOR * (
            actual_home
            -
            expected_home
        )

        elo_ratings[home] += (
            elo_change
        )

        elo_ratings[away] -= (
            elo_change
        )

        # ====================================================
        # UPDATE OVERALL HISTORY
        # ====================================================

        overall_history[home].append({

            "points":
                home_points,

            "win":
                home_win,

            "goals_for":
                home_goals,

            "goals_against":
                away_goals
        })

        overall_history[away].append({

            "points":
                away_points,

            "win":
                away_win,

            "goals_for":
                away_goals,

            "goals_against":
                home_goals
        })

        # ====================================================
        # UPDATE HOME HISTORY
        # ====================================================

        home_history[home].append({

            "win":
                home_win,

            "goals_for":
                home_goals,

            "goals_against":
                away_goals
        })

        # ====================================================
        # UPDATE AWAY HISTORY
        # ====================================================

        away_history[away].append({

            "win":
                away_win,

            "goals_for":
                away_goals,

            "goals_against":
                home_goals
        })


    # ========================================================
    # OVERALL STATISTICS
    # ========================================================

    def overall_stats(history):

        if len(history) == 0:

            return {

                "points": 0.0,

                "win_rate": 0.0,

                "goals_for": 0.0,

                "goals_against": 0.0,

                "goal_difference": 0.0
            }

        n = len(history)

        points = sum(
            item["points"]
            for item in history
        )

        wins = sum(
            item["win"]
            for item in history
        )

        goals_for = sum(
            item["goals_for"]
            for item in history
        )

        goals_against = sum(
            item["goals_against"]
            for item in history
        )

        return {

            "points":
                points / n,

            "win_rate":
                wins / n,

            "goals_for":
                goals_for / n,

            "goals_against":
                goals_against / n,

            "goal_difference":
                (
                    goals_for
                    -
                    goals_against
                ) / n
        }


    # ========================================================
    # HOME / AWAY STATISTICS
    # ========================================================

    def location_stats(history):

        if len(history) == 0:

            return {

                "win_rate": 0.0,

                "goals_for": 0.0,

                "goals_against": 0.0,

                "goal_difference": 0.0
            }

        n = len(history)

        wins = sum(
            item["win"]
            for item in history
        )

        goals_for = sum(
            item["goals_for"]
            for item in history
        )

        goals_against = sum(
            item["goals_against"]
            for item in history
        )

        return {

            "win_rate":
                wins / n,

            "goals_for":
                goals_for / n,

            "goals_against":
                goals_against / n,

            "goal_difference":
                (
                    goals_for
                    -
                    goals_against
                ) / n
        }


    # ========================================================
    # CREATE CURRENT TEAM FEATURES
    # ========================================================

    current_features = {}

    all_teams = (
        set(matches["home_team"])
        |
        set(matches["away_team"])
    )

    for team in all_teams:

        current_features[team] = {

            "overall":
                overall_stats(
                    overall_history[team]
                ),

            "home":
                location_stats(
                    home_history[team]
                ),

            "away":
                location_stats(
                    away_history[team]
                ),

            "elo":
                elo_ratings[team]
        }

    return (
        current_features,
        dict(elo_ratings)
    )


# ============================================================
# CALCULATE FEATURES
# ============================================================

current_features, current_elo = (
    calculate_current_features(
        matches
    )
)


# ============================================================
# CREATE 21 MODEL FEATURES
# ============================================================

def create_match_features(
    home_team,
    away_team
):

    if home_team not in current_features:

        return None

    if away_team not in current_features:

        return None

    home = current_features[
        home_team
    ]

    away = current_features[
        away_team
    ]

    home_overall = home[
        "overall"
    ]

    away_overall = away[
        "overall"
    ]

    home_location = home[
        "home"
    ]

    away_location = away[
        "away"
    ]

    home_elo = current_elo[
        home_team
    ]

    away_elo = current_elo[
        away_team
    ]

    # --------------------------------------------------------
    # Build exact 21 features used by the trained model
    # --------------------------------------------------------

    features = {

        "home_recent_points":
            home_overall["points"],

        "away_recent_points":
            away_overall["points"],

        "home_recent_win_rate":
            home_overall["win_rate"],

        "away_recent_win_rate":
            away_overall["win_rate"],

        "home_recent_goals_for":
            home_overall["goals_for"],

        "away_recent_goals_for":
            away_overall["goals_for"],

        "home_recent_goals_against":
            home_overall["goals_against"],

        "away_recent_goals_against":
            away_overall["goals_against"],

        "home_recent_goal_difference":
            home_overall["goal_difference"],

        "away_recent_goal_difference":
            away_overall["goal_difference"],

        "home_home_win_rate":
            home_location["win_rate"],

        "away_away_win_rate":
            away_location["win_rate"],

        "home_home_goals_for":
            home_location["goals_for"],

        "away_away_goals_for":
            away_location["goals_for"],

        "home_home_goals_against":
            home_location["goals_against"],

        "away_away_goals_against":
            away_location["goals_against"],

        "home_home_goal_difference":
            home_location["goal_difference"],

        "away_away_goal_difference":
            away_location["goal_difference"],

        "home_elo":
            home_elo,

        "away_elo":
            away_elo,

        "elo_difference":
            (
                home_elo
                +
                60
                -
                away_elo
            )
    }

    feature_df = pd.DataFrame(
        [features]
    )

    return feature_df[
        feature_columns
    ]


# ============================================================
# PREDICT MATCH
# ============================================================

def predict_match(
    home_team,
    away_team
):

    feature_df = create_match_features(
        home_team,
        away_team
    )

    if feature_df is None:

        return (
            None,
            None
        )

    scaled_features = (
        scaler.transform(
            feature_df
        )
    )

    prediction = model.predict(
        scaled_features
    )[0]

    probabilities = model.predict_proba(
        scaled_features
    )[0]

    probability_dict = dict(
        zip(
            model.classes_,
            probabilities
        )
    )

    return (
        prediction,
        probability_dict
    )


# ============================================================
# TEAM LIST
# ============================================================

teams = sorted(
    set(
        matches["home_team"]
    )
    |
    set(
        matches["away_team"]
    )
)


CITY = "Manchester City FC (ENG)"

BARCELONA = "FC Barcelona (ESP)"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Match Settings"
)

st.sidebar.caption(
    "Choose the teams for your prediction."
)


# ------------------------------------------------------------
# Quick preset
# ------------------------------------------------------------

preset = st.sidebar.selectbox(

    "Quick Match",

    [
        "Custom",
        "Manchester City vs Barcelona",
        "Barcelona vs Manchester City"
    ]
)


# ------------------------------------------------------------
# Default home / away selections
# ------------------------------------------------------------

if preset == "Manchester City vs Barcelona":

    default_home = (
        teams.index(CITY)
        if CITY in teams
        else 0
    )

    default_away = (
        teams.index(BARCELONA)
        if BARCELONA in teams
        else 0
    )

elif preset == "Barcelona vs Manchester City":

    default_home = (
        teams.index(BARCELONA)
        if BARCELONA in teams
        else 0
    )

    default_away = (
        teams.index(CITY)
        if CITY in teams
        else 0
    )

else:

    default_home = (
        teams.index(CITY)
        if CITY in teams
        else 0
    )

    default_away = (
        teams.index(BARCELONA)
        if BARCELONA in teams
        else 0
    )


# ------------------------------------------------------------
# Team selectors
# ------------------------------------------------------------

home_team = st.sidebar.selectbox(

    "🏠 Home Team",

    teams,

    index=default_home,

    key="main_home_team"
)


away_team = st.sidebar.selectbox(

    "✈️ Away Team",

    teams,

    index=default_away,

    key="main_away_team"
)


# ------------------------------------------------------------
# Model details
# ------------------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader(
    "🤖 Model"
)

st.sidebar.write(
    "Logistic Regression"
)

st.sidebar.write(
    "21 engineered features"
)

st.sidebar.write(
    "Recent Form + Home/Away Form + Elo"
)


show_model_info = st.sidebar.checkbox(
    "Show model information"
)


if show_model_info:

    st.sidebar.info(
        """
        Final Model

        Logistic Regression

        C = 0.01

        Class Weight = balanced

        Training:
        2023/24 + 2024/25

        Test:
        2025/26

        Accuracy:
        44.44%

        Balanced Accuracy:
        42.96%

        Macro F1:
        40.53%
        """
    )


# ============================================================
# MAIN PAGE
# ============================================================

st.title(
    "⚽ UEFA Champions League Predictor"
)

st.subheader(
    "Machine Learning Match Prediction"
)

st.write(
    """
Predict Champions League match outcomes using
recent team form, home/away performance,
goal statistics and Elo team ratings.
"""
)


# ============================================================
# SELECTED MATCH
# ============================================================

st.divider()

st.subheader(
    "⚽ Selected Match"
)

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        f"### 🏠 {home_team}"
    )


with col2:

    st.markdown(
        f"### ✈️ {away_team}"
    )


# ============================================================
# MAIN MATCH PREDICTION
# ============================================================

if home_team == away_team:

    st.warning(
        "Please select two different teams."
    )

else:

    prediction, probabilities = (
        predict_match(
            home_team,
            away_team
        )
    )

    if prediction is not None:

        st.divider()

        st.subheader(
            "🏆 Match Prediction"
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if prediction == "Home Win":

            st.success(
                f"🏠 {home_team} is predicted to win"
            )

        elif prediction == "Away Win":

            st.success(
                f"✈️ {away_team} is predicted to win"
            )

        else:

            st.info(
                "🤝 Draw is predicted"
            )


        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        p1, p2, p3 = st.columns(3)


        with p1:

            st.metric(
                "🏠 Home Win",
                f"{probabilities.get('Home Win', 0) * 100:.1f}%"
            )


        with p2:

            st.metric(
                "🤝 Draw",
                f"{probabilities.get('Draw', 0) * 100:.1f}%"
            )


        with p3:

            st.metric(
                "✈️ Away Win",
                f"{probabilities.get('Away Win', 0) * 100:.1f}%"
            )


        # ----------------------------------------------------
        # Probability chart
        # ----------------------------------------------------

        st.subheader(
            "📊 Model Probabilities"
        )

        probability_df = pd.DataFrame({

            "Outcome": [
                "Home Win",
                "Draw",
                "Away Win"
            ],

            "Probability": [

                probabilities.get(
                    "Home Win",
                    0
                ),

                probabilities.get(
                    "Draw",
                    0
                ),

                probabilities.get(
                    "Away Win",
                    0
                )
            ]
        })


        st.bar_chart(
            probability_df.set_index(
                "Outcome"
            )
        )


        # ----------------------------------------------------
        # Disclaimer
        # ----------------------------------------------------

        st.info(
            """
            ⚠️ This is an ML-based prediction,
            not a guaranteed result. The model achieved
            moderate performance on the held-out 2025/26
            season.
            """
        )


# ============================================================
# TWO-LEGGED KNOCKOUT PREDICTOR
# ============================================================

st.divider()

st.header(
    "🏆 Two-Legged Knockout Predictor"
)

st.write(
    """
Predict both legs of a two-legged Champions League
knockout tie.
"""
)


# ============================================================
# LEG 1
# ============================================================

st.subheader(
    "🥅 Leg 1"
)

leg1_col1, leg1_col2 = st.columns(2)


with leg1_col1:

    leg1_home = st.selectbox(

        "Leg 1 — Home Team",

        teams,

        index=(
            teams.index(BARCELONA)
            if BARCELONA in teams
            else 0
        ),

        key="leg1_home"
    )


with leg1_col2:

    leg1_away = st.selectbox(

        "Leg 1 — Away Team",

        teams,

        index=(
            teams.index(CITY)
            if CITY in teams
            else 0
        ),

        key="leg1_away"
    )


# ============================================================
# LEG 2
# ============================================================

st.subheader(
    "🥅 Leg 2"
)

leg2_col1, leg2_col2 = st.columns(2)


with leg2_col1:

    leg2_home = st.selectbox(

        "Leg 2 — Home Team",

        teams,

        index=(
            teams.index(CITY)
            if CITY in teams
            else 0
        ),

        key="leg2_home"
    )


with leg2_col2:

    leg2_away = st.selectbox(

        "Leg 2 — Away Team",

        teams,

        index=(
            teams.index(BARCELONA)
            if BARCELONA in teams
            else 0
        ),

        key="leg2_away"
    )


# ============================================================
# PREDICT BOTH LEGS
# ============================================================

if st.button(
    "🔮 Predict Both Legs",
    use_container_width=True
):

    if (
        leg1_home == leg1_away
        or
        leg2_home == leg2_away
    ):

        st.warning(
            "Each leg must contain two different teams."
        )

    else:

        pred1, prob1 = predict_match(
            leg1_home,
            leg1_away
        )

        pred2, prob2 = predict_match(
            leg2_home,
            leg2_away
        )

        # ====================================================
        # LEG 1
        # ====================================================

        st.divider()

        st.subheader(
            "🥅 Leg 1 Prediction"
        )

        if pred1 == "Home Win":

            st.success(
                f"🏠 Predicted winner: {leg1_home}"
            )

        elif pred1 == "Away Win":

            st.success(
                f"✈️ Predicted winner: {leg1_away}"
            )

        else:

            st.info(
                "🤝 Predicted draw"
            )


        l1a, l1b, l1c = st.columns(3)


        with l1a:

            st.metric(
                "Home Win",
                f"{prob1.get('Home Win', 0) * 100:.1f}%"
            )


        with l1b:

            st.metric(
                "Draw",
                f"{prob1.get('Draw', 0) * 100:.1f}%"
            )


        with l1c:

            st.metric(
                "Away Win",
                f"{prob1.get('Away Win', 0) * 100:.1f}%"
            )


        # ====================================================
        # LEG 2
        # ====================================================

        st.subheader(
            "🥅 Leg 2 Prediction"
        )

        if pred2 == "Home Win":

            st.success(
                f"🏠 Predicted winner: {leg2_home}"
            )

        elif pred2 == "Away Win":

            st.success(
                f"✈️ Predicted winner: {leg2_away}"
            )

        else:

            st.info(
                "🤝 Predicted draw"
            )


        l2a, l2b, l2c = st.columns(3)


        with l2a:

            st.metric(
                "Home Win",
                f"{prob2.get('Home Win', 0) * 100:.1f}%"
            )


        with l2b:

            st.metric(
                "Draw",
                f"{prob2.get('Draw', 0) * 100:.1f}%"
            )


        with l2c:

            st.metric(
                "Away Win",
                f"{prob2.get('Away Win', 0) * 100:.1f}%"
            )


        # ====================================================
        # TIE INTERPRETATION
        # ====================================================

        st.divider()

        st.subheader(
            "🏆 Tie Interpretation"
        )

        # Team winning both legs
        if (
            pred1 == "Home Win"
            and
            pred2 == "Away Win"
            and
            leg1_home == leg2_away
        ):

            st.success(
                f"""
                🏆 Both predictions favor
                **{leg1_home}**.

                The model therefore favors
                {leg1_home} to progress.
                """
            )

        elif (
            pred1 == "Away Win"
            and
            pred2 == "Home Win"
            and
            leg1_away == leg2_home
        ):

            st.success(
                f"""
                🏆 Both predictions favor
                **{leg1_away}**.

                The model therefore favors
                {leg1_away} to progress.
                """
            )

        else:

            st.warning(
                """
                ⚖️ The two match predictions are mixed.

                A definite aggregate winner cannot be
                calculated because this model predicts
                match outcomes rather than exact scores.

                Therefore, this result should be treated
                as an ML-based indication rather than a
                guaranteed qualification prediction.
                """
            )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()

with st.expander(
    "📘 About this ML Project"
):

    st.write(
        """
        This project uses historical UEFA Champions League
        matches to predict match outcomes.

        The model uses 21 engineered features:

        • Recent points
        • Recent win rate
        • Goals scored
        • Goals conceded
        • Goal difference
        • Home-specific form
        • Away-specific form
        • Home Elo rating
        • Away Elo rating
        • Elo difference

        Final model:

        Logistic Regression

        Training seasons:

        2023/24 and 2024/25

        Held-out test season:

        2025/26

        Accuracy:

        44.44%

        Balanced Accuracy:

        42.96%

        Macro F1:

        40.53%

        The model demonstrates a complete football
        machine-learning pipeline, but its predictions
        should not be interpreted as guaranteed results.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚽ UEFA Champions League Predictor | "
    "Machine Learning Project | "
    "Logistic Regression + Elo + Recent Form"
)