import streamlit as st
import numpy as np
import joblib
import json
import requests
from datetime import date, timedelta
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nepal Disaster Risk Predictor",
    page_icon="🏔️",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #0d1f2d 0%, #1a3a4a 60%, #0d1f2d 100%);
    border: 1px solid #1e4060;
    border-radius: 16px;
    padding: 32px 28px 24px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '🏔️';
    position: absolute;
    right: 24px; top: 20px;
    font-size: 3.5rem;
    opacity: 0.18;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,180,216,0.15);
    border: 1px solid rgba(0,180,216,0.3);
    color: #00b4d8;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 500;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.6rem;
    letter-spacing: 2px;
    color: #fff;
    line-height: 1.05;
    margin-bottom: 8px;
}
.hero-title span { color: #00b4d8; }
.hero-sub { font-size: 0.83rem; color: #7d9bae; font-weight: 300; line-height: 1.5; }

.section-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #00b4d8;
    font-weight: 500;
    margin: 22px 0 10px;
    border-left: 3px solid #00b4d8;
    padding-left: 10px;
}

/* Weather card grid */
.weather-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 12px 0;
}
.weather-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 12px;
    text-align: center;
}
.wc-icon { font-size: 1.4rem; margin-bottom: 4px; }
.wc-val { font-size: 1.3rem; font-weight: 500; color: #e6edf3; }
.wc-label { font-size: 0.7rem; color: #7d9bae; margin-top: 2px; }

/* Result cards */
.result-safe {
    background: linear-gradient(135deg, #0d2b1a, #1a4a2e);
    border: 2px solid #2d6a4f;
    border-radius: 14px; padding: 28px 24px;
    text-align: center; margin-top: 20px;
}
.result-flood {
    background: linear-gradient(135deg, #2b0d0d, #4a1a1a);
    border: 2px solid #e63946;
    border-radius: 14px; padding: 28px 24px;
    text-align: center; margin-top: 20px;
}
.result-land {
    background: linear-gradient(135deg, #2b1a0d, #4a2e00);
    border: 2px solid #f4a261;
    border-radius: 14px; padding: 28px 24px;
    text-align: center; margin-top: 20px;
}
.result-icon { font-size: 3rem; margin-bottom: 6px; }
.result-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem; letter-spacing: 2px; margin-bottom: 4px;
}
.c-safe { color: #52b788; }
.c-flood { color: #e63946; }
.c-land { color: #f4a261; }
.result-sub { font-size: 0.82rem; color: #9db8c8; font-weight: 300; }

/* Confidence bars */
.conf-row { display: flex; align-items: center; gap: 10px; margin: 7px 0; }
.conf-label { font-size: 0.78rem; color: #9db8c8; width: 90px; text-align: right; flex-shrink: 0; }
.conf-bg { flex: 1; background: rgba(255,255,255,0.07); border-radius: 6px; height: 9px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 6px; }
.conf-pct { font-size: 0.78rem; color: #e6edf3; width: 42px; flex-shrink: 0; font-weight: 500; }

.info-box {
    background: rgba(0,180,216,0.07);
    border: 1px solid rgba(0,180,216,0.2);
    border-radius: 10px; padding: 12px 16px;
    font-size: 0.81rem; color: #9db8c8;
    margin-top: 14px; line-height: 1.6;
}
.info-box strong { color: #00b4d8; }

.warn-box {
    background: rgba(244,162,97,0.08);
    border: 1px solid rgba(244,162,97,0.25);
    border-radius: 10px; padding: 12px 16px;
    font-size: 0.81rem; color: #c9a87c;
    margin: 10px 0; line-height: 1.6;
}

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stDateInput"] label {
    font-size: 0.82rem !important;
    color: #9db8c8 !important;
    font-weight: 400 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #7d9bae;
    border-radius: 7px;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #0096c7 !important;
    color: white !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0096c7, #00b4d8) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important; letter-spacing: 2px !important;
    padding: 14px 0 !important; width: 100% !important;
    margin-top: 6px !important;
}
.stButton > button:hover { opacity: 0.87 !important; }

.footer {
    text-align: center; font-size: 0.7rem; color: #3d5a6a;
    margin-top: 40px; padding-top: 14px;
    border-top: 1px solid #1e2d3a;
}
</style>
""", unsafe_allow_html=True)

# ── District coordinates ───────────────────────────────────────────────────────
DISTRICT_COORDS = {
    'achham': (29.0833, 81.3500), 'arghakhanchi': (27.9500, 83.1500),
    'baglung': (28.2667, 83.5833), 'baitadi': (29.5333, 80.4167),
    'bajhang': (29.5500, 81.1833), 'bajura': (29.3333, 81.5500),
    'banke': (28.0500, 81.6167), 'bara': (27.0167, 85.0167),
    'bardiya': (28.3167, 81.4833), 'bhaktapur': (27.6710, 85.4298),
    'bhojpur': (27.1667, 87.0500), 'chitwan': (27.5291, 84.3542),
    'dadeldhura': (29.2967, 80.5786), 'dailekh': (28.8458, 81.7125),
    'dang': (28.0500, 82.3000), 'darchula': (29.8500, 80.5500),
    'dhading': (27.8667, 84.9167), 'dhankuta': (26.9833, 87.3500),
    'dhanusa': (26.8333, 85.9167), 'dolakha': (27.6667, 86.0833),
    'dolpa': (29.0000, 82.9667), 'doti': (29.2667, 80.9500),
    'gorkha': (28.3333, 84.6333), 'gulmi': (28.0667, 83.2667),
    'humla': (29.9667, 81.9833), 'ilam': (26.9167, 87.9167),
    'jajarkot': (28.7000, 82.1833), 'jhapa': (26.6333, 87.8667),
    'jumla': (29.2833, 82.1833), 'kailali': (28.7167, 80.9000),
    'kalikot': (29.1333, 81.6167), 'kanchanpur': (28.8500, 80.3333),
    'kapilbastu': (27.5667, 83.0500), 'kaski': (28.2667, 83.9667),
    'kathmandu': (27.7172, 85.3240), 'kavrepalanchok': (27.5333, 85.6833),
    'khotang': (27.1667, 86.8333), 'lalitpur': (27.6588, 85.3247),
    'lamjung': (28.2167, 84.3833), 'mahottari': (26.6500, 85.7833),
    'makwanpur': (27.4333, 85.0333), 'manang': (28.6667, 84.0167),
    'morang': (26.6500, 87.4500), 'mugu': (29.6833, 82.4333),
    'mustang': (28.9833, 83.8500), 'myagdi': (28.4833, 83.5333),
    'nawalparasi east': (27.5500, 84.3833), 'nawalparasi west': (27.7000, 83.7333),
    'nuwakot': (27.9167, 85.1667), 'okhaldhunga': (27.3167, 86.5000),
    'palpa': (27.8667, 83.5500), 'panchthar': (27.1333, 87.7833),
    'parbat': (28.2167, 83.7000), 'parsa': (27.1000, 84.9833),
    'pyuthan': (28.1000, 82.8667), 'ramechhap': (27.3333, 86.0833),
    'rasuwa': (28.1500, 85.3333), 'rautahat': (27.0000, 85.3000),
    'rolpa': (28.3500, 82.6500), 'rukum east': (28.6167, 82.6500),
    'rukum west': (28.5500, 82.3500), 'rupandehi': (27.5000, 83.4500),
    'salyan': (28.3667, 82.1667), 'sankhuwasabha': (27.3500, 87.3000),
    'saptari': (26.6667, 86.7167), 'sarlahi': (27.0000, 85.5667),
    'sindhuli': (27.2500, 85.9667), 'sindhupalchok': (27.9500, 85.6833),
    'siraha': (26.6500, 86.2000), 'solukhumbu': (27.6667, 86.6667),
    'sunsari': (26.7167, 87.1667), 'surkhet': (28.6000, 81.6167),
    'syangja': (28.0833, 83.8833), 'tanahu': (27.9333, 84.2333),
    'taplejung': (27.3500, 87.6667), 'terhathum': (27.1167, 87.5500),
    'udayapur': (26.9167, 86.5000),
}

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── Load model assets ──────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model   = joblib.load('nepal_disaster_model.pkl')
    scaler  = joblib.load('feature_scaler.pkl')
    encoder = joblib.load('district_label_encoder.pkl')
    with open('feature_columns.json') as f:
        features = json.load(f)
    with open('district_list.json') as f:
        districts = json.load(f)
    return model, scaler, encoder, features, districts

model, scaler, encoder, features, districts = load_assets()

# ── Helper: feature engineering ───────────────────────────────────────────────
def engineer_features(district, precipitation, r3, r7, r30, temp, temp7, month):
    is_monsoon = 1 if 6 <= month <= 9 else 0
    if   6 <= month <= 9:  season = 2
    elif 3 <= month <= 5:  season = 1
    elif month in [10,11]: season = 3
    else:                  season = 0
    dist_enc = encoder.transform([district])[0]
    return np.array([[precipitation, r3, r7, r30, temp, temp7,
                      month, is_monsoon, season, dist_enc]])

# ── Helper: run prediction ─────────────────────────────────────────────────────
def run_prediction(district, precipitation, r3, r7, r30, temp, temp7, month):
    X = engineer_features(district, precipitation, r3, r7, r30, temp, temp7, month)
    X_scaled = scaler.transform(X)
    pred  = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    return pred, proba

# ── Helper: render result card + confidence bars ───────────────────────────────
def render_result(pred, proba, district, month):
    p0, p1, p2 = proba[0]*100, proba[1]*100, proba[2]*100
    season_name = ['Winter','Pre-Monsoon','Monsoon','Post-Monsoon'][
        2 if 6<=month<=9 else (1 if 3<=month<=5 else (3 if month in [10,11] else 0))]
    mn = MONTH_NAMES[month-1]

    if pred == 0:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-icon">🟢</div>
            <div class="result-title c-safe">NO SIGNIFICANT RISK</div>
            <div class="result-sub">{district.title()} · {mn} · {season_name}<br>
            Conditions appear stable. Continue monitoring rainfall.</div>
        </div>""", unsafe_allow_html=True)
    elif pred == 1:
        st.markdown(f"""
        <div class="result-flood">
            <div class="result-icon">🔴</div>
            <div class="result-title c-flood">FLOOD RISK DETECTED</div>
            <div class="result-sub">{district.title()} · {mn} · {season_name}<br>
            Avoid low-lying riverbanks and waterways immediately.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-land">
            <div class="result-icon">🟠</div>
            <div class="result-title c-land">LANDSLIDE RISK DETECTED</div>
            <div class="result-sub">{district.title()} · {mn} · {season_name}<br>
            Avoid steep slopes, hill roads, and river valleys.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">📊 Confidence Breakdown</div>', unsafe_allow_html=True)
    bars = [("No Event", p0, "#52b788"), ("Flood", p1, "#e63946"), ("Landslide", p2, "#f4a261")]
    html = ""
    for label, pct, color in bars:
        html += f"""<div class="conf-row">
            <div class="conf-label">{label}</div>
            <div class="conf-bg"><div class="conf-fill" style="width:{pct:.1f}%;background:{color};"></div></div>
            <div class="conf-pct">{pct:.1f}%</div></div>"""
    st.markdown(html, unsafe_allow_html=True)

    tip = "<strong>Monsoon season active.</strong> ~80% of Nepal's annual rainfall falls Jun–Sep. Risk is naturally elevated." \
        if 6 <= month <= 9 else \
        "<strong>Outside monsoon season.</strong> Risk generally lower, though localised events can still occur in hill districts."
    st.markdown(f'<div class="info-box">ℹ️ {tip}</div>', unsafe_allow_html=True)

# ── Helper: fetch weather from Open-Meteo ─────────────────────────────────────
def fetch_weather(district, past_days=30, forecast_days=16):
    lat, lon = DISTRICT_COORDS[district]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "Asia/Kathmandu"
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()['daily']
    df = pd.DataFrame({
        'date':    pd.to_datetime(data['time']),
        'precip':  [v if v is not None else 0.0 for v in data['precipitation_sum']],
        'tmax':    [v if v is not None else np.nan for v in data['temperature_2m_max']],
        'tmin':    [v if v is not None else np.nan for v in data['temperature_2m_min']],
    })
    df['temp'] = (df['tmax'] + df['tmin']) / 2
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">🇳🇵 Nepal · ML Early Warning System</div>
    <div class="hero-title">DISASTER <span>RISK</span> PREDICTOR</div>
    <div class="hero-sub">
        Flood &amp; Landslide risk assessment · 77 districts · Powered by Open-Meteo &amp; NASA data
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🌧️ Current Risk", "🔮 Future Forecast", "🧪 What-If Simulator"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CURRENT RISK (auto-fetch last 30 days)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-label">📍 Select District</div>', unsafe_allow_html=True)
    district_1 = st.selectbox("District", options=districts,
                               index=districts.index('kathmandu') if 'kathmandu' in districts else 0,
                               key="d1")

    if st.button("FETCH WEATHER & ASSESS RISK", key="btn1"):
        with st.spinner("Fetching live weather data from Open-Meteo..."):
            try:
                df_w = fetch_weather(district_1, past_days=30, forecast_days=1)
                today_row = df_w[df_w['date'] <= pd.Timestamp.now()].iloc[-1]
                today_idx = df_w[df_w['date'] <= pd.Timestamp.now()].index[-1]

                precip_today = float(df_w.loc[today_idx, 'precip'])
                r3  = float(df_w.loc[max(0, today_idx-2):today_idx, 'precip'].sum())
                r7  = float(df_w.loc[max(0, today_idx-6):today_idx, 'precip'].sum())
                r30 = float(df_w.loc[max(0, today_idx-29):today_idx, 'precip'].sum())
                temp     = float(df_w.loc[today_idx, 'temp'])
                temp_7d  = float(df_w.loc[max(0, today_idx-6):today_idx, 'temp'].mean())
                month    = int(today_row['date'].month)

                # Show fetched values
                st.markdown('<div class="section-label">📡 Auto-Fetched Weather Data</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="weather-grid">
                    <div class="weather-card">
                        <div class="wc-icon">🌧️</div>
                        <div class="wc-val">{precip_today:.1f}</div>
                        <div class="wc-label">Today (mm)</div>
                    </div>
                    <div class="weather-card">
                        <div class="wc-icon">📅</div>
                        <div class="wc-val">{r3:.1f}</div>
                        <div class="wc-label">3-Day Total (mm)</div>
                    </div>
                    <div class="weather-card">
                        <div class="wc-icon">🗓️</div>
                        <div class="wc-val">{r7:.1f}</div>
                        <div class="wc-label">7-Day Total (mm)</div>
                    </div>
                    <div class="weather-card">
                        <div class="wc-icon">📆</div>
                        <div class="wc-val">{r30:.1f}</div>
                        <div class="wc-label">30-Day Total (mm)</div>
                    </div>
                    <div class="weather-card">
                        <div class="wc-icon">🌡️</div>
                        <div class="wc-val">{temp:.1f}°</div>
                        <div class="wc-label">Temp Today (°C)</div>
                    </div>
                    <div class="weather-card">
                        <div class="wc-icon">📊</div>
                        <div class="wc-val">{temp_7d:.1f}°</div>
                        <div class="wc-label">7-Day Avg Temp</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                pred, proba = run_prediction(district_1, precip_today, r3, r7, r30,
                                             temp, temp_7d, month)
                render_result(pred, proba, district_1, month)

            except Exception as e:
                st.error(f"Could not fetch weather data: {e}\n\nCheck your internet connection or try the What-If Simulator tab.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — FUTURE FORECAST (up to 16 days ahead)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label">📍 Select District & Date</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])
    with col_a:
        district_2 = st.selectbox("District", options=districts,
                                   index=districts.index('kathmandu') if 'kathmandu' in districts else 0,
                                   key="d2")
    with col_b:
        max_date = date.today() + timedelta(days=15)
        forecast_date = st.date_input("Forecast Date",
                                       value=date.today() + timedelta(days=3),
                                       min_value=date.today() + timedelta(days=1),
                                       max_value=max_date)

    st.markdown('<div class="warn-box">⚡ Forecasts beyond 7 days have lower accuracy. Use as a general risk guide, not an exact prediction.</div>',
                unsafe_allow_html=True)

    if st.button("FETCH FORECAST & ASSESS RISK", key="btn2"):
        with st.spinner("Fetching weather forecast from Open-Meteo..."):
            try:
                days_ahead = (forecast_date - date.today()).days
                df_w2 = fetch_weather(district_2, past_days=30, forecast_days=days_ahead+1)

                target_dt = pd.Timestamp(forecast_date)
                idx = df_w2[df_w2['date'] == target_dt].index
                if len(idx) == 0:
                    st.error("No forecast data available for that date. Try a closer date.")
                else:
                    i = idx[0]
                    precip_f = float(df_w2.loc[i, 'precip'])
                    r3_f  = float(df_w2.loc[max(0,i-2):i, 'precip'].sum())
                    r7_f  = float(df_w2.loc[max(0,i-6):i, 'precip'].sum())
                    r30_f = float(df_w2.loc[max(0,i-29):i, 'precip'].sum())
                    temp_f    = float(df_w2.loc[i, 'temp'])
                    temp7_f   = float(df_w2.loc[max(0,i-6):i, 'temp'].mean())
                    month_f   = int(target_dt.month)

                    st.markdown('<div class="section-label">📡 Forecast Weather Data</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="weather-grid">
                        <div class="weather-card">
                            <div class="wc-icon">🌧️</div>
                            <div class="wc-val">{precip_f:.1f}</div>
                            <div class="wc-label">That Day (mm)</div>
                        </div>
                        <div class="weather-card">
                            <div class="wc-icon">📅</div>
                            <div class="wc-val">{r3_f:.1f}</div>
                            <div class="wc-label">Prior 3 Days (mm)</div>
                        </div>
                        <div class="weather-card">
                            <div class="wc-icon">🗓️</div>
                            <div class="wc-val">{r7_f:.1f}</div>
                            <div class="wc-label">Prior 7 Days (mm)</div>
                        </div>
                        <div class="weather-card">
                            <div class="wc-icon">📆</div>
                            <div class="wc-val">{r30_f:.1f}</div>
                            <div class="wc-label">Prior 30 Days (mm)</div>
                        </div>
                        <div class="weather-card">
                            <div class="wc-icon">🌡️</div>
                            <div class="wc-val">{temp_f:.1f}°</div>
                            <div class="wc-label">Forecast Temp (°C)</div>
                        </div>
                        <div class="weather-card">
                            <div class="wc-icon">📊</div>
                            <div class="wc-val">{temp7_f:.1f}°</div>
                            <div class="wc-label">7-Day Avg Temp</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    pred2, proba2 = run_prediction(district_2, precip_f, r3_f, r7_f,
                                                    r30_f, temp_f, temp7_f, month_f)
                    render_result(pred2, proba2, district_2, month_f)

            except Exception as e:
                st.error(f"Could not fetch forecast: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — WHAT-IF SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="info-box">
    🧪 <strong>What-If Simulator</strong> — manually enter any weather scenario to see predicted risk.
    Useful for testing "what if it rains 150mm over 3 days?" without needing real data.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">📍 Location & Time</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        district_3 = st.selectbox("District", options=districts,
                                   index=districts.index('sindhupalchok') if 'sindhupalchok' in districts else 0,
                                   key="d3")
    with col2:
        month_3 = st.selectbox("Month", list(range(1,13)),
                                format_func=lambda m: MONTH_NAMES[m-1], index=7, key="m3")

    st.markdown('<div class="section-label">🌧️ Rainfall Scenario</div>', unsafe_allow_html=True)
    st.caption("💡 Light rain: 0–10mm/day · Moderate: 10–50mm · Heavy: 50–100mm · Extreme: 100mm+")

    col3, col4 = st.columns(2)
    with col3:
        p3       = st.number_input("Today's Rainfall (mm)", 0.0, 500.0, 80.0, 1.0, key="p3")
        r7_3     = st.number_input("Last 7 Days Total (mm)", 0.0, 1000.0, 200.0, 5.0, key="r7_3")
    with col4:
        r3_3     = st.number_input("Last 3 Days Total (mm)", 0.0, 600.0, 180.0, 5.0, key="r3_3")
        r30_3    = st.number_input("Last 30 Days Total (mm)", 0.0, 2000.0, 450.0, 10.0, key="r30_3")

    st.markdown('<div class="section-label">🌡️ Temperature</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        temp_3   = st.number_input("Temperature (°C)", -10.0, 45.0, 22.0, 0.5, key="t3")
    with col6:
        temp7_3  = st.number_input("7-Day Avg Temp (°C)", -10.0, 45.0, 21.0, 0.5, key="t73")

    if st.button("RUN SIMULATION", key="btn3"):
        pred3, proba3 = run_prediction(district_3, p3, r3_3, r7_3, r30_3,
                                        temp_3, temp7_3, month_3)
        render_result(pred3, proba3, district_3, month_3)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Logistic Regression · 210,197 training records (1982–2019) · NASA POWER + Open-Meteo<br>
    77 Nepali districts · Built for educational purposes · Not a substitute for official warnings
</div>
""", unsafe_allow_html=True)
