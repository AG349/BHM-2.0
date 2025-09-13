import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# ML & utils (optional)
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -------------------- GENERIC SMS ALERT FUNCTION --------------------
def send_sms_alert(to_number, message):
    # Placeholder - replace with actual SMS API later (Fast2SMS, msg91, AWS SNS, etc.)
    st.info(f"📩 SMS sent to {to_number}: {message}")
    return True

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="BHOOMI Rockfall AI - Enhanced", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    body { background-color: #0d1117; color: #00FFEF; }
    .stMetric { background: rgba(0, 255, 239, 0.1);
                border-radius: 15px; padding: 10px; border: 1px solid #00FFEF; }
    .stDataFrame { border: 1px solid #00FFEF; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 BHOOMI Safety Interface (Enhanced)")
st.markdown("### AI-Powered Rockfall Prediction & Alert System")
st.markdown("System Status: 🔵 Online | Mode: Multimodal Fusion Active")
st.divider()

# -------------------- AUTH (simple role) --------------------
with st.sidebar:
    st.header("User")
    username = st.text_input("Username", value="guest")
    password = st.text_input("Password (demo)", type="password")
    role = st.selectbox("Role", ["viewer","supervisor","admin"]) if username and password else "viewer"

# -------------------- DATA SOURCE --------------------
mode = st.radio("📊 Select Data Source:", ["Simulated Live Data", "Preloaded CSV", "Upload CSV"]) 

# helper to ensure dataframe has correct dtypes
def sanitize_df(df):
    if 'Timestamp' in df.columns:
        try:
            df['ts_parsed'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        except Exception:
            df['ts_parsed'] = pd.NaT
    else:
        df['ts_parsed'] = pd.NaT
    for col in ['Vibration','Slope','Risk']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

if mode == "Preloaded CSV":
    try:
        df = pd.read_csv("mine_sensor_data.csv")
        st.success("✅ Preloaded CSV loaded successfully!")
        df = sanitize_df(df)
    except FileNotFoundError:
        st.error("⚠ Preloaded file 'mine_sensor_data.csv' not found.")
        st.stop()

elif mode == "Upload CSV":
    uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"]) 
    if uploaded:
        df = pd.read_csv(uploaded)
        df = sanitize_df(df)
        st.success("✅ Uploaded CSV loaded successfully!")
    else:
        st.warning("Please upload a CSV to continue.")
        st.stop()

else:
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["Timestamp","Vibration","Slope","Weather","Risk"]) 
        for i in range(30):
            ts = (datetime.now() - timedelta(seconds=(30-i)*30)).strftime("%H:%M:%S")
            st.session_state.df.loc[len(st.session_state.df)] = [ts, round(np.random.normal(0.5,0.2),3), round(np.random.normal(45,3),2), np.random.choice(["Sunny","Rainy","Cloudy","Windy"]), np.random.randint(10,90)]
    new_data = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Vibration": float(round(np.random.normal(0.5,0.2),3)),
        "Slope": float(round(np.random.normal(45,3),2)),
        "Weather": np.random.choice(["Sunny","Rainy","Cloudy","Windy"]),
        "Risk": int(np.random.randint(0,100))
    }
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_data])], ignore_index=True)
    st.session_state.df = st.session_state.df.tail(500).reset_index(drop=True)
    df = st.session_state.df.copy()

if 'Risk' not in df.columns:
    df['Risk'] = np.random.randint(0,100,size=len(df))

# -------------------- DATA ENHANCEMENTS --------------------
if 'Vibration' in df.columns:
    df['Vibration_MA_5'] = df['Vibration'].rolling(5, min_periods=1).mean()
if 'Slope' in df.columns:
    df['Slope_MA_5'] = df['Slope'].rolling(5, min_periods=1).mean()

def composite_index(row):
    v = (row.get('Vibration',0) - df['Vibration'].min()) / max(1e-6,(df['Vibration'].max()-df['Vibration'].min())) if 'Vibration' in row else 0
    s = (row.get('Slope',0) - df['Slope'].min()) / max(1e-6,(df['Slope'].max()-df['Slope'].min())) if 'Slope' in row else 0
    r = row.get('Risk',0)/100
    return round(100*(0.5*r + 0.3*v + 0.2*s),2)

df['CompositeIndex'] = df.apply(composite_index, axis=1)

# -------------------- METRICS --------------------
col1, col2, col3, col4 = st.columns(4)
current_risk = int(df['Risk'].iloc[-1])
if current_risk > 70:
    risk_status = "🔴 HIGH"
elif current_risk > 40:
    risk_status = "🟡 MEDIUM"
else:
    risk_status = "🟢 LOW"

with col1: st.metric("Current Risk", risk_status)
with col2: st.metric("Active Sensors", "📸 5 | 🎙 3")
with col3: st.metric("Last Update", str(df['Timestamp'].iloc[-1]))
with col4: st.metric("Weather", df['Weather'].iloc[-1])

st.divider()


# -------------------- DYNAMIC RISK GAUGE --------------------
st.subheader("🧭 Risk Gauge")
risk_min, risk_max = df['Risk'].min(), df['Risk'].max()
risk_range = max(1, risk_max - risk_min)
risk_low = risk_min + 0.3 * risk_range
risk_high = risk_min + 0.7 * risk_range

gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=current_risk,
    title={"text":"Current Risk %"},
    gauge={
        "axis":{"range":[0,100]},
        "bar":{"color":"cyan"},
        "steps":[
            {"range":[0,risk_low],"color":"green"},
            {"range":[risk_low,risk_high],"color":"yellow"},
            {"range":[risk_high,100],"color":"red"}
        ]
    }
))
gauge.update_layout(paper_bgcolor="#0d1117", font={"color":"#00FFEF"})
st.plotly_chart(gauge, use_container_width=True)

# -------------------- VIBRATION + SLOPE (Dynamic Zones) --------------------
col_a, col_b = st.columns(2)

# --- Vibration ---
vib_min, vib_max = df["Vibration"].min(), df["Vibration"].max()
vib_range = vib_max - vib_min
vib_low = vib_min + 0.3 * vib_range
vib_high = vib_min + 0.7 * vib_range

with col_a:
    st.subheader("📈 Vibration Trend")
    fig_vibration = px.line(df, x="Timestamp", y="Vibration", markers=True,
                            title="Vibration Levels", line_shape="spline",
                            color_discrete_sequence=["orange"])
    fig_vibration.update_layout(template="plotly_dark",
                                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
    fig_vibration.add_hrect(y0=vib_min, y1=vib_low, fillcolor="green", opacity=0.2, line_width=0, annotation_text="Low", annotation_position="left")
    fig_vibration.add_hrect(y0=vib_high, y1=vib_max, fillcolor="red", opacity=0.2, line_width=0, annotation_text="High", annotation_position="left")
    st.plotly_chart(fig_vibration, use_container_width=True)

# --- Slope ---
slope_min, slope_max = df["Slope"].min(), df["Slope"].max()
slope_range = slope_max - slope_min
slope_low = slope_min + 0.3 * slope_range
slope_high = slope_min + 0.7 * slope_range

with col_b:
    st.subheader("⛰ Slope Angle Trend")
    fig_slope = px.line(df, x="Timestamp", y="Slope", markers=True,
                        title="Slope Angle", line_shape="spline",
                        color_discrete_sequence=["lime"])
    fig_slope.update_layout(template="plotly_dark",
                            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
    fig_slope.add_hrect(y0=slope_min, y1=slope_low, fillcolor="green", opacity=0.2, line_width=0, annotation_text="Low", annotation_position="left")
    fig_slope.add_hrect(y0=slope_high, y1=slope_max, fillcolor="red", opacity=0.2, line_width=0, annotation_text="High", annotation_position="left")
    st.plotly_chart(fig_slope, use_container_width=True)


# -------------------- THERMAL HEATMAP --------------------
st.subheader("🌡 Thermal Heatmap with Sensor Hotspots")
heat_data = np.random.normal(loc=current_risk, scale=15, size=(20, 20))
heat_data = np.clip(heat_data, 0, 100)

heat_fig = px.imshow(
    heat_data,
    color_continuous_scale="plasma",
    origin="lower",
    aspect="auto",
    labels=dict(color="Temperature / Risk Level"),
    title="Thermal Activity Heatmap",
    zmin=0, zmax=100
)

sensor_x = np.random.randint(0, 20, 6)
sensor_y = np.random.randint(0, 20, 6)
heat_fig.add_trace(go.Scatter(
    x=sensor_x, y=sensor_y,
    mode="markers+text",
    marker=dict(size=12, color="white", symbol="x"),
    text=[f"Sensor {i+1}" for i in range(6)],
    textposition="top center"
))

low_threshold = np.percentile(heat_data, 30)
high_threshold = np.percentile(heat_data, 70)
heat_fig.add_hrect(y0=0, y1=low_threshold, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Low Risk", annotation_position="bottom left")
heat_fig.add_hrect(y0=high_threshold, y1=100, fillcolor="red", opacity=0.1, line_width=0, annotation_text="High Risk", annotation_position="top left")

heat_fig.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
st.plotly_chart(heat_fig, use_container_width=True)

# -------------------- ANOMALY DETECTION --------------------
st.subheader("🔎 Anomaly Detection (IsolationForest)")
if SKLEARN_AVAILABLE and 'Vibration' in df.columns and len(df.dropna())>10:
    try:
        iso = IsolationForest(contamination=0.05, random_state=42)
        features = df[['Vibration','Slope','Risk']].fillna(method='ffill').fillna(0)
        iso.fit(features)
        df['anomaly_score'] = iso.decision_function(features)
        df['anomaly'] = iso.predict(features)
        anomalies = df[df['anomaly']==-1]
        st.write(f"Detected {len(anomalies)} anomalies")
        if not anomalies.empty:
            st.dataframe(anomalies[['Timestamp','Vibration','Slope','Risk','anomaly_score']].tail(10))
    except Exception as e:
        st.warning(f"Anomaly detection failed: {e}")
else:
    st.info("scikit-learn not available or insufficient data — skipping anomaly detection")

st.divider()

# -------------------- RESTRICTED AREA & WORKER GEO --------------------
st.subheader("🚫 Restricted Area Detection")
restricted_areas = ["Zone A", "Zone C", "Zone E"]
worker_zones = np.random.choice(["Zone A","Zone B","Zone C","Zone D","Zone E"], size=5)
restricted_alerts = [zone for zone in worker_zones if zone in restricted_areas]

if restricted_alerts:
    st.warning(f"⚠ Restricted Area Alert! Workers detected in: {', '.join(restricted_alerts)}")
    alerts.loc[len(alerts)] = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Vibration": np.nan,
        "Slope": np.nan,
        "Weather": np.nan,
        "Risk": 100,
        "Action": "🚫 Restricted Area Entry"
    }
else:
    st.info("✅ No workers in restricted areas.")

mine_center = {"lat": 20.5937, "lon": 78.9629}
num_workers = 10
worker_positions = pd.DataFrame({
    "Worker": [f"Worker {i+1}" for i in range(num_workers)],
    "lat": mine_center["lat"] + np.random.uniform(-0.01, 0.01, num_workers),
    "lon": mine_center["lon"] + np.random.uniform(-0.01, 0.01, num_workers)
})

restricted_zone = {"lat": mine_center["lat"] + 0.005,
                   "lon": mine_center["lon"] - 0.005,
                   "radius_km": 0.7}

fig_workers = px.scatter_mapbox(
    worker_positions, lat="lat", lon="lon", text="Worker",
    zoom=14, height=600, color_discrete_sequence=["cyan"]
)
fig_workers.add_trace(go.Scattermapbox(
    lat=[restricted_zone["lat"]],
    lon=[restricted_zone["lon"]],
    mode="markers+text",
    marker=dict(size=18, color="red"),
    text=["🚫 Restricted Zone"],
    textposition="top right"
))
fig_workers.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0d1117", font=dict(color="white"))
st.plotly_chart(fig_workers, use_container_width=True)

if st.button("📢 Alert Workers Near Restricted Area"):
    if restricted_alerts:
        st.success(f"✅ Alert sent to workers in restricted zones: {', '.join(restricted_alerts)}")
    else:
        st.info("ℹ No workers currently near restricted areas to alert.")

# -------------------- ALERT SYSTEM (Manual + Restricted) --------------------
st.subheader("🚨 Alerts System")

# Restricted Area Alerts
restricted_areas = ["Zone A", "Zone C", "Zone E"]
worker_zones = np.random.choice(["Zone A","Zone B","Zone C","Zone D","Zone E"], size=5)
restricted_alerts = [zone for zone in worker_zones if zone in restricted_areas]

if restricted_alerts:
    st.warning(f"⚠ Restricted Area Alert! Workers detected in: {', '.join(restricted_alerts)}")
    send_sms_alert("+911234567890", f"Restricted Area Alert! Workers in {', '.join(restricted_alerts)}")
else:
    st.info("✅ No workers in restricted areas.")

# Manual Alert
if st.button("🚨 SEND MANUAL ALERT"):
    send_sms_alert("+911234567890", "🚨 Manual Alert triggered from BHOOMI system")
    st.success("✅ Manual alert sent!")

st.divider()

# -------------------- FORECAST & PREDICTION --------------------
st.subheader("🔮 Forecast (Next 6 Hours)")
hours = [f"{i}h" for i in range(1,7)]
forecast = np.random.randint(20,95,size=6)
df_forecast = pd.DataFrame({"Hour":hours,"Forecast Risk %":forecast})
fig_forecast = px.bar(df_forecast, x="Hour", y="Forecast Risk %",
                      color="Forecast Risk %", title="Predicted Risk Probability",
                      color_continuous_scale="turbo")
fig_forecast.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
st.plotly_chart(fig_forecast, use_container_width=True)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("🧠 BHOOMI Safety Core v3.1 Enhanced | Alerts system includes SMS Stub | TEAM BHOOMI ⚡")

st_autorefresh(interval=60*1000, key="auto_refresh_enhanced")
