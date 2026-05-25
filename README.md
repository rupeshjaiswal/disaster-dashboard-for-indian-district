# DisasterSense – India Flood Alert Dashboard

An AI-powered disaster management dashboard that predicts flood risk across Indian districts using Machine Learning, Weather APIs, GIS Mapping, FastAPI, and Streamlit.

## Features
- Flood Risk Prediction using XGBoost
- Real-time Weather Monitoring
- Interactive GIS Risk Map
- Disaster Alerts Dashboard
- Risk Score Visualization
- District-wise Analysis

## Tech Stack
- Python
- FastAPI
- Streamlit
- XGBoost
- Pandas
- Plotly
- Open-Meteo API

<img width="1918" height="906" alt="image" src="https://github.com/user-attachments/assets/be91a5e9-9b8a-4dc6-9e31-b7d13f996ab8" />
  <img width="1918" height="907" alt="image" src="https://github.com/user-attachments/assets/cbad6033-58fc-4d4a-a2f1-48439a246200" />


## Installation

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload

Cd Frontend
pip install -r requirements.txt
python -m streamlit run app.py

