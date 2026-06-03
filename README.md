Full-Stack World Cup Simulation Engine
An end-to-end Machine Learning microservice and web application designed to predict the outcomes of the 2026 FIFA World Cup.

Unlike standard binary win/loss predictors, this engine utilizes an XGBoost classifier to output exact match probabilities (Win/Draw/Loss), calculates Expected Points (xPts), and mathematically simulates the entire 72-match group stage to generate real-time tournament standings.

🚀 Key Features
Dynamic Feature Engineering: Instead of relying on static data, the backend dynamically calculates historical ELO ratings, recent goal form (last 5 matches), and schedule congestion (rest days) on the fly for any requested matchup.

Optimized Draw Probabilities: The XGBoost model utilizes custom class weights to accurately predict international football's most difficult outcome: the draw.

FastAPI Microservice: The ML model is serialized and served via a lightning-fast, asynchronous Python API, completely decoupling the prediction engine from the frontend UI.

Tournament Simulator: A custom Python simulation loop that processes the entire World Cup schedule, aggregates Expected Points, and mathematically determines the knockout stage qualifiers.

💻 Tech Stack
Machine Learning & Data Engineering

Algorithm: XGBoost Classifier

Data Manipulation: Pandas, NumPy

Serialization: Joblib

Backend & MLOps

Framework: FastAPI, Uvicorn

Deployment: Docker, Google Cloud Run (Planned)

Frontend (Work in Progress)

Framework: Next.js (App Router), React

Styling: Tailwind CSS

🏗️ Architecture
Plaintext
worldcup-backend/
├── data/                    # Master schedule and historical ELO/Results CSVs
├── services/                
│   └── feature_builder.py   # Dynamically calculates form and ELO differentials
├── main.py                  # FastAPI routing and expected points (xPts) logic
└── xgboost_worldcup.pkl     # Serialized prediction model
🔌 API Reference
1. Head-to-Head Predictor
Simulates a single match between any two teams and returns exact outcome probabilities.

Endpoint: POST /api/head-to-head

Payload: ```json
{
"home_team": "Mexico",
"away_team": "South Africa",
"is_neutral": 0
}


2. Group Stage Simulator
Loops through the official 72-match schedule, calculates xPts for every team, and returns the sorted standings for all 12 groups.

Endpoint: GET /api/simulate/groups

⚙️ Getting Started (Local Development)
Prerequisites
Python 3.10+

pip

Installation & Setup
Clone the repository

Bash
git clone https://github.com/[Your-Username]/world-cup-predictor.git
cd world-cup-predictor/worldcup-backend
Install dependencies

Bash
pip install -r requirements.txt
Run the FastAPI server

Bash
uvicorn main:app --reload
Test the API
Navigate to http://localhost:8000/docs in your browser to interact with the auto-generated Swagger UI and test the endpoints.

🗺️ Roadmap
[x] Train baseline Random Forest and XGBoost models

[x] Engineer dynamic rolling features (ELO, Goal Form, Rest Days)

[x] Build FastAPI backend and serialize model

[x] Create Expected Points (xPts) group stage simulation loop

[ ] Containerize API with Docker

[ ] Build interactive Next.js dashboard for Head-to-Head and Group Tables

[ ] Deploy full-stack application to the cloud
