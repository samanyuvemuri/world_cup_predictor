# 🌍 FIFA World Cup 2026 Simulation Engine

An end-to-end Machine Learning microservice and web application designed to predict the outcomes of the **2026 FIFA World Cup**.

Unlike traditional binary win/loss predictors, this engine leverages an **XGBoost classifier** to generate exact match outcome probabilities (**Win / Draw / Loss**), calculate **Expected Points (xPts)**, and mathematically simulate the entire tournament group stage to produce projected standings and qualifiers.

---

## 🚀 Features

### ⚽ Match Outcome Prediction

Predicts the probability of each possible match result:

* Home Win
* Draw
* Away Win

Returns calibrated probabilities instead of a single prediction.

### 📊 Dynamic Feature Engineering

Rather than relying on static precomputed features, the backend dynamically generates match-specific inputs:

* Historical ELO Ratings
* Recent Goal Form (last 5 matches)
* Schedule Congestion / Rest Days
* ELO Differentials
* Form Differentials

This allows the API to generate predictions for any requested matchup in real time.

### 🎯 Optimized Draw Prediction

International football draws are notoriously difficult to model.

The XGBoost classifier utilizes custom class weighting and probability calibration techniques to improve draw prediction performance while maintaining overall model accuracy.

### ⚡ FastAPI Prediction Service

The trained model is serialized with Joblib and served through a high-performance FastAPI microservice.

Benefits include:

* Asynchronous request handling
* Decoupled ML architecture
* Easy frontend integration
* Production-ready deployment

### 🏆 Tournament Simulator

A custom simulation engine processes the official World Cup group-stage schedule:

* Simulates all 72 group-stage matches
* Calculates Expected Points (xPts)
* Aggregates team performance
* Generates projected group standings
* Determines likely knockout-stage qualifiers

---

## 💻 Tech Stack

### Machine Learning & Data Engineering

| Component           | Technology         |
| ------------------- | ------------------ |
| Algorithm           | XGBoost Classifier |
| Data Processing     | Pandas             |
| Numerical Computing | NumPy              |
| Model Serialization | Joblib             |

### Backend & MLOps

| Component        | Technology                 |
| ---------------- | -------------------------- |
| API Framework    | FastAPI                    |
| ASGI Server      | Uvicorn                    |
| Containerization | Docker (Planned)           |
| Cloud Deployment | Google Cloud Run (Planned) |

### Frontend (Work in Progress)

| Component  | Technology           |
| ---------- | -------------------- |
| Framework  | Next.js (App Router) |
| UI Library | React                |
| Styling    | Tailwind CSS         |

---

## 🏗️ Project Architecture

```text
worldcup-backend/
│
├── data/
│   ├── schedule.csv
│   ├── elo_ratings.csv
|   ├── groups.json
│   └── historical_results.csv
│
├── services/
│   └── feature_builder.py
│
├── training/
│   ├── randomforest_model.py
│   └── xgboost_model.py
│
├── main.py
│
├── xgboost_worldcup_model.pkl
│
├── Dockerfile
│
└── requirements.txt
```

### Directory Breakdown

#### `data/`

Contains:

* Historical international match results
* ELO ratings
* Official World Cup schedule

#### `services/feature_builder.py`

Responsible for dynamically generating:

* ELO differentials
* Goal form metrics
* Rest day calculations
* Match-specific features

#### `main.py`

Handles:

* FastAPI routing
* Prediction requests
* Expected Points (xPts) calculations
* Tournament simulations

#### `xgboost_worldcup.pkl`

Serialized XGBoost model used by the API.

---

## 🔌 API Reference

### Head-to-Head Predictor

Simulates a single match between any two teams and returns outcome probabilities.

**Endpoint**

```http
POST /api/head-to-head
```

**Request Body**

```json
{
  "home_team": "United States",
  "away_team": "Paraguay",
  "is_neutral": 0
}
```

**Example Response**

```json
{
  "home_team": "United States",
  "away_team": "Paraguay",
  "probabilities": {
    "away_win": 0.279,
    "draw": 0.385,
    "home_win": 0.336
  }
}
```

---

### Group Stage Simulator

Simulates the entire World Cup group stage and generates projected standings.

**Endpoint**

```http
GET /api/simulate/groups
```

**Example Response**

```json
{
  "A": {
    "Mexico": {
      "xPts": 6.24,
      "matches_played": 3
    },
    "South Africa": {
      "xPts": 1.62,
      "matches_played": 3
    },
  }
}
```

---

## ⚙️ Local Development Setup

### Prerequisites

* Python 3.10+
* pip

---

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/world-cup-predictor.git

cd world-cup-predictor/worldcup-backend
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI Server

```bash
uvicorn main:app --reload
```

### 4. Test the API

Open your browser and navigate to:

```text
http://localhost:8000/docs
```

FastAPI will automatically generate interactive Swagger documentation for testing endpoints.

---

## 📈 Machine Learning Pipeline

### Data Sources

* Historical international match results
* FIFA/ELO rating datasets
* Tournament schedule data

### Engineered Features

* Team ELO Rating
* ELO Differential
* Goals Scored (Last 5 Matches)
* Goals Conceded (Last 5 Matches)
* Goal Difference Form
* Rest Days
* Neutral Venue Indicator

### Model Selection

Models evaluated:

* Logistic Regression
* Random Forest
* XGBoost

Final model:

✅ **XGBoost Classifier**

Selected for superior multiclass classification performance and probability calibration.

---

## 🗺️ Roadmap

### Completed

* [x] Train various models to determine the most effective one
* [x] Engineer dynamic rolling features
* [x] Build FastAPI backend
* [x] Serialize model with Joblib
* [x] Implement Expected Points (xPts) calculations
* [x] Create full group-stage simulation engine

### In Progress

* [ ] Docker containerization
* [ ] Interactive Next.js dashboard
* [ ] Tournament visualization UI
* [ ] Cloud deployment on Google Cloud Run
* [ ] Monte Carlo tournament simulations
* [ ] Knockout-stage probability projections

---
