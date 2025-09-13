import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from twilio.rest import Client

# -------------------- TWILIO CONFIG --------------------
# ⚠ Replace these with your real Twilio credentials
TWILIO_ACCOUNT_SID = "AC085550d3a2de2449dd757374516d5fee"
TWILIO_AUTH_TOKEN = "e8a3c375557824bd19cdf01b8f6fbb2b"
TWILIO_FROM_NUMBER = "+18053077157"   # Twilio verified number

# SMS sending function
def send_sms_alert(to_number, message):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_FROM_NUMBER,
            to=to_number
        )
        st.success(f"📩 SMS sent to {to_number}")
        return True
    except Exception as e:
        st.error(f"❌ SMS failed: {e}")
        return False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="BHOOMI Rockfall AI", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    body { background-color: #0d1117; color: #00FFEF; }
    .stMetric { background: rgba(0, 255, 239, 0.1);
                border-radius: 15px; padding: 10px; border: 1px solid #00FFEF; }
    .stDataFrame { border: 1px solid #00FFEF; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 BHOOMI Safety Interface")
st.markdown("### AI-Powered Rockfall Prediction & Alert System")
st.markdown("System Status: 🔵 Online | Mode: Multimodal Fusion Active")
st.divider()

# -------------------- SIDEBAR CONFIG --------------------
st.sidebar.header("📱 SMS Alert Settings")
recipient_number = st.sidebar.text_input("Enter Phone Number (+91...)", "")

# -------------------- RISK CALCULATION FUNCTION --------------------
def calculate_risk(vibration, slope, weather, heat_data):
    # Normalize vibration (0–2g → 0–100 scale)
    vib_factor = min(max(vibration / 2 * 100, 0), 100)

    # Normalize slope (0–90° → 0–100 scale)
    slope_factor = min(max(slope / 90 * 100, 0), 100)

    # Weather impact
    weather_factor = {
        "Sunny": 20,
        "Cloudy": 40,
        "Windy": 60,
        "Rainy": 80
    }.get(weather, 30)

    # Heatmap avg (center 10x10 region of 20x20 grid)
    center_heat = heat_data[5:15, 5:15].mean()

    # Weighted risk calculation
    risk_score = (
        0.4 * vib_factor +
        0.3 * slope_factor +
        0.2 * weather_factor +
        0.1 * center_heat
    )

    return round(min(risk_score, 100), 2)

# -------------------- DATA SOURCE --------------------
mode = st.radio("📊 Select Data Source:", ["Simulated Live Data", "Preloaded CSV", "Upload CSV"])

if mode == "Preloaded CSV":
    try:
        df = pd.read_csv("mine_sensor_data.csv")
        st.success("✅ Preloaded CSV loaded successfully!")
    except:
        st.error("⚠ Preloaded file 'mine_sensor_data.csv' not found.")
        st.stop()

elif mode == "Upload CSV":
    uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("✅ Uploaded CSV loaded successfully!")
    else:
        st.warning("Please upload a CSV to continue.")
        st.stop()

else:  # Simulated Live Data
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["Timestamp","Vibration","Slope","Weather","Risk"])

    # Generate simulated sensor data
    heat_data = np.random.normal(loc=50, scale=20, size=(20, 20))
    heat_data = np.clip(heat_data, 0, 100)

    new_data = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Vibration": round(np.random.normal(0.5, 0.2), 3),
        "Slope": round(np.random.normal(45, 3), 2),
        "Weather": np.random.choice(["Sunny", "Rainy", "Cloudy", "Windy"]),
    }
    new_data["Risk"] = calculate_risk(new_data["Vibration"], new_data["Slope"], new_data["Weather"], heat_data)

    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_data])], ignore_index=True)
    df = st.session_state.df.tail(50)

# -------------------- METRICS --------------------
col1, col2, col3, col4 = st.columns(4)
current_risk = df["Risk"].iloc[-1]
if current_risk > 70:
    risk_status = "🔴 HIGH"
elif current_risk > 40:
    risk_status = "🟡 MEDIUM"
else:
    risk_status = "🟢 LOW"

with col1: st.metric("Current Risk", risk_status)
with col2: st.metric("Active Sensors", "📸 5 | 🎙 3")
with col3: st.metric("Last Update", str(df["Timestamp"].iloc[-1]))
with col4: st.metric("Weather", df["Weather"].iloc[-1])

st.divider()

# -------------------- DYNAMIC RISK GAUGE --------------------
st.subheader("🧭 Risk Gauge")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=current_risk,
    title={"text": "Current Risk %"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "cyan"},
        "steps": [
            {"range": [0, 40], "color": "green"},
            {"range": [40, 70], "color": "yellow"},
            {"range": [70, 100], "color": "red"}
        ]
    }
))

gauge.update_layout(
    paper_bgcolor="#0d1117",
    font={"color": "#00FFEF"}
)

st.plotly_chart(gauge, use_container_width=True)

# -------------------- VIBRATION + SLOPE --------------------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📈 Vibration Trend")
    fig_vibration = px.line(df, x="Timestamp", y="Vibration", markers=True,
                            title="Vibration Levels", line_shape="spline",
                            color_discrete_sequence=["orange"])
    fig_vibration.update_layout(template="plotly_dark",
                                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
    st.plotly_chart(fig_vibration, use_container_width=True)

with col_b:
    st.subheader("⛰ Slope Angle Trend")
    fig_slope = px.line(df, x="Timestamp", y="Slope", markers=True,
                        title="Slope Angle", line_shape="spline",
                        color_discrete_sequence=["lime"])
    fig_slope.update_layout(template="plotly_dark",
                            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
    st.plotly_chart(fig_slope, use_container_width=True)

# -------------------- THERMAL HEATMAP WITH SENSORS + AXES --------------------
st.subheader("🌡 Thermal Heatmap with Sensors")

# Sensor positions (X=0–20, Y=0–20 since grid is 20x20)
sensors = {
    "S1": (3, 15),
    "S2": (5, 12),
    "S3": (16, 5),
    "S4": (18, 14),
    "S5": (10, 8),
    "S6": (14, 6),
}

heat_fig = go.Figure(data=go.Heatmap(
    z=heat_data,
    colorscale="Viridis",
    zmin=0, zmax=100,
    colorbar=dict(
        title="Temperature/Risk Level",
        tickvals=[0, 50, 100],
        ticktext=["--0 Low ", "--50 Medium", "--100 High"]
    )
))

# Add sensors inside heatmap
for name, (x, y) in sensors.items():
    heat_fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode="markers+text",
        marker=dict(size=12, color="white", symbol="x"),
        text=[name],
        textposition="top center",
        showlegend=False
    ))

# ✅ Show axes with labels
heat_fig.update_layout(
    title="Thermal Activity Heatmap",
    template="plotly_dark",
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    xaxis=dict(title="X Axis", range=[0, 20], showgrid=True, zeroline=False),
    yaxis=dict(title="Y Axis", range=[0, 20], showgrid=True, zeroline=False),
    height=600
)

st.plotly_chart(heat_fig, use_container_width=True)

# -------------------- ALERTS LOG --------------------
st.subheader("🚨 Alerts Log")
alerts = df.tail(5).copy()
alerts["Action"] = np.where(alerts["Risk"]>70,"🔴 Evacuation",
                     np.where(alerts["Risk"]>40,"🟡 Warning","🟢 Monitoring"))
st.dataframe(alerts, use_container_width=True)

# -------------------- AUTO REFRESH --------------------
st_autorefresh(interval=60*1000, key="auto_refresh")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("🧠 BHOOMI Safety Core v3.2 | Live + CSV + Alerts + Forecast + Heatmap + GeoMap | TEAM BHOOMI ⚡")
