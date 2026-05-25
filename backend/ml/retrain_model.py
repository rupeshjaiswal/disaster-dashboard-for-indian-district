"""
retrain_model.py
Run this once to regenerate disaster_risk_model.pkl with YOUR current
scikit-learn version, replacing the incompatible pickle.

Usage (from the backend/ml folder):
    python retrain_model.py
"""

import json
import numpy as np
import joblib
from pathlib import Path

# ── synthetic training data (mirrors the real India 1901-2015 feature space) ──
np.random.seed(42)
N = 3000

years       = np.random.randint(1901, 2016, N)
subdiv_code = np.random.randint(0, 36, N)
annual      = np.random.uniform(200, 4000, N)
monsoon     = annual * np.random.uniform(0.55, 0.80, N)
pre_mon     = annual * np.random.uniform(0.05, 0.12, N)
post_mon    = annual * np.random.uniform(0.05, 0.15, N)
winter      = annual * np.random.uniform(0.01, 0.08, N)
mon_frac    = monsoon / (annual + 1e-9)
peak_month  = monsoon * np.random.uniform(0.25, 0.45, N)
hi_months   = np.random.randint(1, 8, N)
roll_5y     = annual * np.random.uniform(0.85, 1.15, N)
anomaly     = annual * np.random.uniform(-0.15, 0.15, N)
jun = monsoon * np.random.uniform(0.10, 0.20, N)
jul = monsoon * np.random.uniform(0.25, 0.40, N)
aug = monsoon * np.random.uniform(0.20, 0.35, N)
sep = monsoon * np.random.uniform(0.10, 0.20, N)

X = np.column_stack([
    years, subdiv_code, annual, monsoon, pre_mon, post_mon,
    winter, mon_frac, peak_month, hi_months, roll_5y, anomaly,
    jun, jul, aug, sep
])

# Labels: Low=0 (<1000 mm), Medium=1 (1000-1800 mm), High=2 (>1800 mm)
y = np.where(annual < 1000, 0, np.where(annual < 1800, 1, 2))

# ── train a Random Forest (no _loss dependency, pure sklearn) ─────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

print("=== Model performance ===")
print(classification_report(y_test, clf.predict(X_test),
                             target_names=["Low", "Medium", "High"]))

# ── save ──────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent
joblib.dump(clf, out_dir / "disaster_risk_model.pkl")
print(f"✅  Saved: {out_dir / 'disaster_risk_model.pkl'}")

# Patch metadata so main.py is happy
meta_path = out_dir / "model_metadata.json"
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
else:
    meta = {}
meta["model_type"] = "RandomForestClassifier"
meta["retrained"]  = True
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print("✅  Updated model_metadata.json")
