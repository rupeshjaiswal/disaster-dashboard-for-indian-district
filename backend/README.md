# 🌊 DisasterSense — AI-Based Disaster Alert & Resource Dashboard

A full-stack AI system for flood risk prediction across Indian districts.

---

## 📁 Project Structure

```
disaster-dashboard/
├── notebooks/
│   └── disaster_model_training.ipynb   ← Google Colab training notebook
├── backend/
│   ├── main.py                          ← FastAPI server
│   ├── requirements.txt
│   └── ml/
│       ├── disaster_risk_model.pkl      ← (place here after Colab training)
│       ├── model_metadata.json
│       └── subdivision_codes.json
└── frontend/
    ├── app.py                           ← Streamlit dashboard
    └── requirements.txt
```

---

## 🧠 Dataset Recommendation

**✅ Use: Rainfall in India 1901–2015**
- https://www.kaggle.com/datasets/aravindpcoder/rainfall-in-india-1901-2015/data
- India-specific, 115 years of data, subdivision + monthly + annual breakdown
- Perfect for district-level flood risk modeling

❌ Avoid: Global Natural Calamities — too coarse, lacks rainfall granularity for India

---

## 🚀 Step 1 — Train Model (Google Colab)

1. Open `notebooks/disaster_model_training.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Upload the Kaggle CSV file when prompted
4. Run all cells
5. Download 3 output files:
   - `disaster_risk_model.pkl`
   - `model_metadata.json`
   - `subdivision_codes.json`
6. Place all 3 files in `backend/ml/`

**Algorithm Used:** XGBoost (primary) + Random Forest (compared)
- SMOTE for class imbalance handling
- 5-fold cross-validation
- Features: Annual, Monsoon, Pre/Post monsoon, Anomaly, Rolling 5Y Mean, etc.

**Risk Labels:**
| Code | Label  | Annual Rainfall |
|------|--------|-----------------|
| 0    | Low    | < 1000 mm/yr    |
| 1    | Medium | 1000–1800 mm/yr |
| 2    | High   | > 1800 mm/yr    |

---

## 🚀 Step 2 — Run FastAPI Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: http://localhost:8000
Swagger docs: http://localhost:8000/docs

### Key Endpoints:
| Endpoint | Description |
|----------|-------------|
| `GET /districts` | List all districts |
| `POST /predict` | Predict risk from rainfall data |
| `GET /weather/{district}` | Live weather + forecast |
| `GET /dashboard/{district}` | Full dashboard data |
| `GET /map-data` | Risk levels for all districts |

---

## 🚀 Step 3 — Run Streamlit Dashboard

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Dashboard at: http://localhost:8501

---

## 🗺️ Dashboard Features

### Risk Map (Leaflet)
- Dark basemap (CartoDB Dark Matter)
- Color-coded district markers: 🟢 Safe / 🟡 Medium / 🔴 Danger
- Relief center overlays: 🏥 Medical / 🏠 Shelter / 🍱 Food
- Interactive popups with district details

### Weather Data (Open-Meteo — Free, No API Key!)
- Current temperature & wind speed
- 7-day rainfall forecast
- Annual rainfall estimate

### Alert System
- Automatically generated based on ML prediction
- Severity levels: Info / Warning / Danger
- Live timestamp

### Resource Panel
- Hospitals available
- Shelters available
- Food supply status
- Relief centers list

### Trend Graph
- 7-day bar chart with color coding:
  - 🔵 Normal (<8mm)
  - 🟡 Elevated (8–15mm)
  - 🔴 Heavy (>15mm)

---

## 🌐 Free APIs Used

| API | Usage | Cost |
|-----|-------|------|
| Open-Meteo | Weather + 7-day forecast | FREE, no key |
| CartoDB Tiles | Dark map tiles | FREE |

---

## 📊 Model Architecture

```
Raw Features (16)
    ↓
SMOTE (balance classes)
    ↓
XGBoost Classifier
    ├── n_estimators: 300
    ├── max_depth: 8
    ├── learning_rate: 0.05
    └── subsample: 0.8
    ↓
3-Class Output: Low / Medium / High
```

Expected metrics (typical):
- Accuracy: ~87–93%
- F1 (weighted): ~0.88–0.92
- CV F1: ~0.85–0.91

---

## 🔄 Data Flow

```
User opens dashboard
        ↓
Streamlit fetches /dashboard/{district}
        ↓
FastAPI calls Open-Meteo weather API
        ↓
FastAPI builds feature vector
        ↓
ML model predicts risk (Low/Med/High)
        ↓
Dashboard renders Map + Alerts + Chart + Resources
```

---

## 🧪 Demo Mode

If model `.pkl` files are not present, the backend auto-switches to **demo mode** using threshold rules:
- Annual > 1800mm → High
- Annual > 1000mm → Medium
- Annual ≤ 1000mm → Low

This lets you test the dashboard before training.
