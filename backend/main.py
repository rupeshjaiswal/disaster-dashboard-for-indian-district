"""
Disaster Alert API — FastAPI Backend
Serves the ML model and fetches weather data.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import numpy as np
import httpx
import os
from datetime import datetime, timedelta
from typing import Optional
import asyncio

# ─── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Disaster Alert API",
    description="AI-based disaster risk prediction for Indian districts",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model & Metadata ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "disaster_risk_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "ml", "model_metadata.json")
SUBDIV_PATH = os.path.join(BASE_DIR, "ml", "subdivision_codes.json")

model = None
metadata = {}
subdivision_map = {}

@app.on_event("startup")
async def load_model():
    global model, metadata, subdivision_map
    # Load metadata & subdivision map (non-critical)
    try:
        with open(META_PATH) as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load metadata: {e}")

    try:
        with open(SUBDIV_PATH) as f:
            subdivision_map = json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load subdivision codes: {e}")

    # Load ML model — catch ALL errors so the server always starts
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except ModuleNotFoundError as e:
        print(
            f"⚠️  Model pickle is incompatible with this scikit-learn version:\n"
            f"   {e}\n"
            f"   ➜  Run: python backend/ml/retrain_model.py  to rebuild it.\n"
            f"   Falling back to demo (rule-based) predictions."
        )
        model = None
    except FileNotFoundError:
        print("⚠️  Model file not found — running in demo mode.")
        model = None
    except Exception as e:
        print(f"⚠️  Unexpected error loading model: {e}\n   Falling back to demo mode.")
        model = None

# ─── Districts Data ──────────────────────────────────────────────────────────
DISTRICTS = {
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "subdivision": "West Rajasthan"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "subdivision": "Konkan & Goa"},
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "subdivision": "West Uttar Pradesh"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "subdivision": "Gangetic West Bengal"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "subdivision": "Tamil Nadu"},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946, "subdivision": "Karnataka"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "subdivision": "Telangana"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "subdivision": "Gujarat Region"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "subdivision": "Madhya Maharashtra"},
    "Patna": {"lat": 25.5941, "lon": 85.1376, "subdivision": "Bihar"},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126, "subdivision": "Madhya Pradesh"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "subdivision": "West Uttar Pradesh"},
    "Kochi": {"lat": 9.9312, "lon": 76.2673, "subdivision": "Kerala"},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362, "subdivision": "Assam & Meghalaya"},
    "Bhubaneswar": {"lat": 20.2961, "lon": 85.8245, "subdivision": "Odisha"},
}

# ─── Dummy resource data ─────────────────────────────────────────────────────
RESOURCES = {
    "Jaipur":     {"hospitals": 24, "shelters": 8,  "food_status": "Adequate",   "relief_centers": 5},
    "Mumbai":     {"hospitals": 87, "shelters": 32, "food_status": "Good",       "relief_centers": 18},
    "Delhi":      {"hospitals": 112,"shelters": 45, "food_status": "Good",       "relief_centers": 22},
    "Kolkata":    {"hospitals": 63, "shelters": 28, "food_status": "Adequate",   "relief_centers": 14},
    "Chennai":    {"hospitals": 58, "shelters": 22, "food_status": "Good",       "relief_centers": 11},
    "Bangalore":  {"hospitals": 72, "shelters": 18, "food_status": "Good",       "relief_centers": 9},
    "Hyderabad":  {"hospitals": 55, "shelters": 20, "food_status": "Adequate",   "relief_centers": 10},
    "Ahmedabad":  {"hospitals": 48, "shelters": 15, "food_status": "Low",        "relief_centers": 7},
    "Pune":       {"hospitals": 44, "shelters": 16, "food_status": "Adequate",   "relief_centers": 8},
    "Patna":      {"hospitals": 31, "shelters": 12, "food_status": "Critical",   "relief_centers": 6},
    "Bhopal":     {"hospitals": 29, "shelters": 11, "food_status": "Adequate",   "relief_centers": 5},
    "Lucknow":    {"hospitals": 38, "shelters": 14, "food_status": "Adequate",   "relief_centers": 7},
    "Kochi":      {"hospitals": 41, "shelters": 19, "food_status": "Good",       "relief_centers": 9},
    "Guwahati":   {"hospitals": 22, "shelters": 9,  "food_status": "Low",        "relief_centers": 4},
    "Bhubaneswar":{"hospitals": 26, "shelters": 10, "food_status": "Adequate",   "relief_centers": 5},
}

RELIEF_CENTERS = {
    "Jaipur": [
        {"name": "SMS Hospital Relief",     "lat": 26.9085, "lon": 75.7858, "type": "Medical"},
        {"name": "Adarsh Nagar Shelter",    "lat": 26.9344, "lon": 75.7868, "type": "Shelter"},
        {"name": "Mansarovar Food Camp",    "lat": 26.8548, "lon": 75.7556, "type": "Food"},
    ],
    "Mumbai": [
        {"name": "KEM Hospital Relief",       "lat": 19.0015, "lon": 72.8404, "type": "Medical"},
        {"name": "Dharavi Shelter Camp",      "lat": 19.0413, "lon": 72.8545, "type": "Shelter"},
        {"name": "Andheri Food Distribution", "lat": 19.1136, "lon": 72.8697, "type": "Food"},
    ],
}

# ─── Schemas ─────────────────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    annual_rainfall: float
    monsoon_total: float
    pre_monsoon: float = 0
    post_monsoon: float = 0
    winter: float = 0
    peak_month_rain: float = 0
    high_rain_months: int = 4
    rolling_5y_mean: float = 0
    rainfall_anomaly: float = 0
    year: int = 2024
    subdivision: str = "West Rajasthan"
    jun: float = 0
    jul: float = 0
    aug: float = 0
    sep: float = 0

# ─── Helpers ─────────────────────────────────────────────────────────────────
def _make_features(data: PredictionInput):
    """Build feature vector for model input."""
    subdiv_code = subdivision_map.get(data.subdivision, 0)
    annual = data.annual_rainfall
    monsoon = data.monsoon_total
    monsoon_fraction = monsoon / (annual + 1e-9)

    return np.array([[
        data.year,
        subdiv_code,
        annual,
        monsoon,
        data.pre_monsoon,
        data.post_monsoon,
        data.winter,
        monsoon_fraction,
        data.peak_month_rain or monsoon * 0.35,
        data.high_rain_months,
        data.rolling_5y_mean or annual,
        data.rainfall_anomaly,
        data.jun, data.jul, data.aug, data.sep
    ]])

def _demo_predict(annual_rainfall: float):
    """Fallback if model not loaded."""
    if annual_rainfall > 1800:
        return 2, [0.05, 0.15, 0.80]
    elif annual_rainfall > 1000:
        return 1, [0.15, 0.65, 0.20]
    else:
        return 0, [0.75, 0.20, 0.05]

async def _fetch_open_meteo(lat: float, lon: float):
    """Fetch 7-day rainfall from Open-Meteo (free, no API key needed)."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,weathercode,temperature_2m_max,temperature_2m_min"
        f"&current_weather=true"
        f"&timezone=Asia%2FKolkata"
        f"&forecast_days=7"
    )
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Disaster Alert API is running 🚨",
        "version": "1.0.0",
        "model_loaded": model is not None
    }

@app.get("/districts")
async def get_districts():
    """List all available districts."""
    return {"districts": list(DISTRICTS.keys())}

@app.post("/predict")
async def predict_risk(data: PredictionInput):
    """Predict flood risk from rainfall data."""
    risk_names = {0: "Low", 1: "Medium", 2: "High"}
    risk_colors = {0: "#22c55e", 1: "#eab308", 2: "#ef4444"}

    if model is not None:
        features = _make_features(data)
        label = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0].tolist()
    else:
        label, proba = _demo_predict(data.annual_rainfall)

    return {
        "risk_level": risk_names[label],
        "risk_code": label,
        "risk_color": risk_colors[label],
        "probability": {
            "Low": round(proba[0], 3),
            "Medium": round(proba[1], 3),
            "High": round(proba[2], 3)
        },
        "alert": _generate_alert(label, data.annual_rainfall),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/weather/{district}")
async def get_weather(district: str):
    """Fetch live weather + 7-day rainfall forecast for a district."""
    if district not in DISTRICTS:
        raise HTTPException(status_code=404, detail=f"District '{district}' not found")

    info = DISTRICTS[district]
    try:
        raw = await _fetch_open_meteo(info["lat"], info["lon"])
    except Exception:
        return _demo_weather(district, info)

    daily = raw.get("daily", {})
    current = raw.get("current_weather", {})

    dates = daily.get("time", [])
    precip = daily.get("precipitation_sum", [0]*7)
    temp_max = daily.get("temperature_2m_max", [30]*7)
    temp_min = daily.get("temperature_2m_min", [20]*7)
    weathercodes = daily.get("weathercode", [0]*7)

    total_7day = sum(p or 0 for p in precip)
    annual_est  = total_7day * 52

    return {
        "district": district,
        "lat": info["lat"],
        "lon": info["lon"],
        "current_temp_c": current.get("temperature", 30),
        "current_wind_kmh": current.get("windspeed", 10),
        "is_day": current.get("is_day", 1),
        "forecast_7days": [
            {
                "date": dates[i] if i < len(dates) else "",
                "precipitation_mm": round(precip[i] or 0, 1),
                "temp_max_c": round(temp_max[i] or 30, 1),
                "temp_min_c": round(temp_min[i] or 20, 1),
                "weathercode": weathercodes[i] or 0
            }
            for i in range(min(7, len(dates)))
        ],
        "total_7day_rain_mm": round(total_7day, 1),
        "annual_est_mm": round(annual_est, 1),
        "data_source": "Open-Meteo (free)"
    }

@app.get("/dashboard/{district}")
async def get_dashboard(district: str):
    """Full dashboard data: weather + risk prediction + resources + alerts."""
    if district not in DISTRICTS:
        raise HTTPException(status_code=404, detail=f"District '{district}' not found")

    info = DISTRICTS[district]

    # Fetch weather
    try:
        raw = await _fetch_open_meteo(info["lat"], info["lon"])
        daily = raw.get("daily", {})
        current = raw.get("current_weather", {})
        precip = daily.get("precipitation_sum", [0]*7)
        total_7day = sum(p or 0 for p in precip)
        annual_est = total_7day * 52
    except Exception:
        total_7day = 45.0
        annual_est = 2340.0
        precip = [3.2, 8.5, 12.1, 6.4, 2.8, 9.3, 3.7]
        current = {"temperature": 28, "windspeed": 15}

    # Build prediction input
    pred_input = PredictionInput(
        annual_rainfall=annual_est,
        monsoon_total=annual_est * 0.65,
        pre_monsoon=annual_est * 0.08,
        post_monsoon=annual_est * 0.1,
        winter=annual_est * 0.05,
        peak_month_rain=max(precip) * 30,
        high_rain_months=sum(1 for p in precip if p > 5),
        rolling_5y_mean=annual_est * 0.95,
        rainfall_anomaly=annual_est * 0.05,
        year=datetime.now().year,
        subdivision=info["subdivision"],
        jun=precip[0] * 30 if len(precip) > 0 else 0,
        jul=precip[1] * 30 if len(precip) > 1 else 0,
        aug=precip[2] * 30 if len(precip) > 2 else 0,
        sep=precip[3] * 30 if len(precip) > 3 else 0,
    )

    # Predict
    risk_names = {0: "Low", 1: "Medium", 2: "High"}
    risk_colors = {0: "#22c55e", 1: "#eab308", 2: "#ef4444"}
    if model:
        features = _make_features(pred_input)
        label = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0].tolist()
    else:
        label, proba = _demo_predict(annual_est)

    resources = RESOURCES.get(district, {
        "hospitals": 15, "shelters": 6,
        "food_status": "Adequate", "relief_centers": 4
    })
    relief = RELIEF_CENTERS.get(district, [
        {"name": f"{district} Relief Center", "lat": info["lat"]+0.01,
         "lon": info["lon"]+0.01, "type": "Shelter"}
    ])

    dates_list = []
    for i in range(7):
        d = datetime.now() - timedelta(days=6-i)
        dates_list.append(d.strftime("%b %d"))

    return {
        "district": district,
        "lat": info["lat"],
        "lon": info["lon"],
        "risk": {
            "level": risk_names[label],
            "code": label,
            "color": risk_colors[label],
            "probability": {
                "Low": round(proba[0], 3),
                "Medium": round(proba[1], 3),
                "High": round(proba[2], 3)
            }
        },
        "weather": {
            "current_temp_c": current.get("temperature", 28),
            "current_wind_kmh": current.get("windspeed", 15),
            "total_7day_rain_mm": round(total_7day, 1),
            "annual_est_mm": round(annual_est, 1),
            "forecast": [
                {
                    "date": dates_list[i],
                    "precipitation_mm": round(precip[i] if i < len(precip) else 0, 1)
                }
                for i in range(7)
            ]
        },
        "resources": resources,
        "relief_centers": relief,
        "alerts": _generate_alerts(label, total_7day, district),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/map-data")
async def get_map_data():
    """Return risk levels for all districts for map overlay."""
    results = []
    for district, info in DISTRICTS.items():
        rainfall_typical = {
            "Jaipur": 480,  "Mumbai": 2400, "Delhi": 800,
            "Kolkata": 1800,"Chennai": 1400,"Bangalore": 970,
            "Hyderabad": 810,"Ahmedabad": 800,"Pune": 720,
            "Patna": 1200,  "Bhopal": 1150,"Lucknow": 1020,
            "Kochi": 3100,  "Guwahati": 2800,"Bhubaneswar": 1500
        }
        annual = rainfall_typical.get(district, 1000)

        if model:
            subdiv = info["subdivision"]
            subdiv_code = subdivision_map.get(subdiv, 0)
            features = np.array([[
                2024, subdiv_code, annual, annual*0.65,
                annual*0.08, annual*0.1, annual*0.05, 0.65,
                annual*0.2, 5, annual, 0,
                annual*0.1, annual*0.2, annual*0.18, annual*0.1
            ]])
            label = int(model.predict(features)[0])
        else:
            label, _ = _demo_predict(annual)

        risk_names  = {0: "Low", 1: "Medium", 2: "High"}
        risk_colors = {0: "#22c55e", 1: "#eab308", 2: "#ef4444"}
        results.append({
            "district": district,
            "lat": info["lat"],
            "lon": info["lon"],
            "risk_level": risk_names[label],
            "risk_code": label,
            "color": risk_colors[label],
            "annual_rainfall_mm": annual
        })
    return {"districts": results}

# ─── Internal helpers ─────────────────────────────────────────────────────────

def _generate_alert(label: int, annual: float) -> str:
    if label == 2:
        return f"🔴 FLOOD ALERT: Extremely high rainfall ({annual:.0f}mm/yr). Evacuate low-lying areas."
    elif label == 1:
        return f"🟡 WATCH: Elevated rainfall levels ({annual:.0f}mm/yr). Monitor water bodies."
    else:
        return f"🟢 NORMAL: Rainfall within safe range ({annual:.0f}mm/yr)."

def _generate_alerts(label: int, rain_7day: float, district: str) -> list:
    alerts = []
    now = datetime.now().strftime("%H:%M")
    if label == 2:
        alerts.append({"type": "danger", "icon": "🔴",
            "title": "FLOOD RISK ALERT",
            "message": f"High flood risk detected in {district}. Activate emergency protocols.",
            "time": now})
    if rain_7day > 50:
        alerts.append({"type": "warning", "icon": "⚠️",
            "title": "Heavy Rainfall Warning",
            "message": f"{rain_7day:.1f}mm of rain in last 7 days. River levels may rise.",
            "time": now})
    if label == 1:
        alerts.append({"type": "warning", "icon": "🟡",
            "title": "Flood Watch Active",
            "message": f"Medium risk in {district}. Stay alert and avoid flood-prone areas.",
            "time": now})
    if not alerts:
        alerts.append({"type": "info", "icon": "🟢",
            "title": "All Clear",
            "message": f"No immediate flood threat in {district}.",
            "time": now})
    return alerts

def _demo_weather(district: str, info: dict) -> dict:
    import random
    random.seed(hash(district) % 1000)
    precip = [round(random.uniform(0, 15), 1) for _ in range(7)]
    dates = [(datetime.now() - timedelta(days=6-i)).strftime("%Y-%m-%d") for i in range(7)]
    return {
        "district": district,
        "lat": info["lat"], "lon": info["lon"],
        "current_temp_c": round(random.uniform(22, 36), 1),
        "current_wind_kmh": round(random.uniform(5, 30), 1),
        "is_day": 1,
        "forecast_7days": [{"date": dates[i], "precipitation_mm": precip[i],
                             "temp_max_c": 32, "temp_min_c": 22, "weathercode": 61}
                            for i in range(7)],
        "total_7day_rain_mm": round(sum(precip), 1),
        "annual_est_mm": round(sum(precip) * 52, 1),
        "data_source": "Demo (Open-Meteo unavailable)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
