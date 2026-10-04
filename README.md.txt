# ⚽ UEFA Champions League Match Predictor

A machine learning project that predicts UEFA Champions League match outcomes using historical match data, recent team form, home/away performance, and Elo team ratings.

The project also includes a Streamlit web application for predicting both legs of a two-legged knockout tie.

---

## 🎯 Project Objective

The goal of this project is to build an end-to-end football machine learning system capable of estimating the outcome of UEFA Champions League matches.

The system uses historical Champions League matches to learn patterns related to:

- Recent team performance
- Win rate
- Goals scored and conceded
- Goal difference
- Home-specific performance
- Away-specific performance
- Elo team strength ratings

The final model predicts one of three possible outcomes:

- 🏠 Home Win
- 🤝 Draw
- ✈️ Away Win

---

## 🧠 Machine Learning Pipeline

```text
Historical Match Data
        ↓
Data Cleaning & Parsing
        ↓
Feature Engineering
        ↓
Recent Team Form
        ↓
Home/Away Form
        ↓
Elo Ratings
        ↓
Chronological Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Saved ML Model
        ↓
Streamlit Prediction App