import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Traffic Situation Classifier", page_icon="🚦")

# ==========================================
# 1. LOAD FILES SAVED BY THE NEW NOTEBOOK
# ==========================================
@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.pkl")
    meta = joblib.load("model_meta.pkl")  # feature order + class names
    models = {
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "Logistic Regression": joblib.load("logistic_model.pkl"),
        "Decision Tree": joblib.load("decision_tree_model.pkl"),
    }
    return scaler, meta, models


scaler, meta, models = load_assets()
FEATURES = meta["features"]
CLASS_NAMES = meta["class_names"]  # ["low", "medium", "high"] -> labels 0, 1, 2

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DISPLAY = {
    "low": ("Low Traffic", "🟢", st.success),
    "medium": ("Medium Traffic", "🟡", st.warning),
    "high": ("High Traffic", "🔴", st.error),
}

# Ranges seen in the training data (the model has never seen values outside these)
TRAIN_MAX = {"Car": 180, "Bike": 70, "Bus": 50, "Truck": 60}


# ==========================================
# 2. FEATURE BUILDER (must match the notebook exactly)
# ==========================================
def build_input_row(car, bike, bus, truck, day_name, t):
    minutes = t.hour * 60 + t.minute
    total = car + bike + bus + truck
    day_num = DAYS.index(day_name)
    row = {
        "CarCount": car,
        "BikeCount": bike,
        "BusCount": bus,
        "TruckCount": truck,
        "Total": total,
        "Heavy_Vehicle_Ratio": (bus + truck) / (total + 1e-5),
        "Day_Num": day_num,
        "Is_Weekend": int(day_num >= 5),
        "Time_sin": np.sin(2 * np.pi * minutes / 1440),
        "Time_cos": np.cos(2 * np.pi * minutes / 1440),
    }
    return pd.DataFrame([row])[FEATURES]


# ==========================================
# 3. USER INTERFACE
# ==========================================
st.title("🚦 Traffic Situation Classifier")
st.write("Classify current road conditions from live vehicle counts, day and time.")

st.sidebar.header("Model Configuration")
selected_model_name = st.sidebar.selectbox("Choose Active Classifier", list(models.keys()))
st.sidebar.success(f"Running Inference via: **{selected_model_name}**")

st.header("Current Vehicle Metrics")
col1, col2 = st.columns(2)
car = col1.number_input("Car Count", min_value=0, value=25)
bus = col2.number_input("Bus Count", min_value=0, value=2)
bike = col1.number_input("Bike Count", min_value=0, value=5)
truck = col2.number_input("Truck Count", min_value=0, value=10)
st.caption(f"Total vehicles: **{car + bike + bus + truck}**")

for name, val in {"Car": car, "Bike": bike, "Bus": bus, "Truck": truck}.items():
    if val > TRAIN_MAX[name]:
        st.warning(
            f"{name} count {val} is above the training range (max {TRAIN_MAX[name]}). "
            "The prediction may be unreliable."
        )

st.header("Temporal Parameters")
day = st.selectbox("Day of the Week", DAYS)
time_value = st.time_input("Time of Day", value=pd.Timestamp("12:00").time())

# ==========================================
# 4. PREDICTION
# ==========================================
if st.button("Generate Traffic Prediction"):
    model = models[selected_model_name]
    x = build_input_row(car, bike, bus, truck, day, time_value)
    x_scaled = scaler.transform(x)

    proba = model.predict_proba(x_scaled)[0]
    label = CLASS_NAMES[int(np.argmax(proba))]
    text, icon, show = DISPLAY[label]
    show(f"Prediction: **{text}** {icon}  (confidence {proba.max():.0%})")

    st.write("Class probabilities")
    st.bar_chart(pd.Series(proba, index=[c.capitalize() for c in CLASS_NAMES]))
