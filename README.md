# 🌊 Nepal Disaster Predictor

## 🚀 Live Demo
👉 [Try the app here](https://manishbhatt1-nepal-disaster-predictor.streamlit.app)

A machine learning system that predicts flood and landslide risk across all 77 districts of Nepal — trained on 37 years of historical disaster data and powered by live weather from the OpenMeteo API.

No manual weather input needed. Just select a district, and the app pulls real-time conditions automatically.

---

## Why I built this

Nepal loses dozens of lives every monsoon season to floods and landslides. Most of those losses happen in districts that are already known to be high-risk — but without a fast, accessible tool to communicate that risk in real time, warnings come too late or not at all.

I wanted to see whether historical disaster patterns combined with live weather data could produce something actually useful — not just a notebook that scores well on a test set, but a working app that anyone could open and check.

---

## How it works

```
Historical disaster data (37 years, all 77 districts)
        +
Live weather data (OpenMeteo API — auto-fetched by district)
        ↓
ML Model (trained on rainfall, temperature, seasonal patterns)
        ↓
Risk prediction: Flood risk / Landslide risk for selected district
        ↓
Streamlit web app — interactive, no technical knowledge required
```

---

## Features

- Covers **all 77 districts** of Nepal
- **37 years** of historical flood and landslide records used for training
- **Live weather fetched automatically** via OpenMeteo API — no manual input
- Full **Streamlit web interface** — select a district, get a risk assessment
- Separate notebooks for model training and rainfall data enrichment

---

## Project Structure

```
Nepal-Disaster-Predictor/
│
├── Nepal_Model_Training.ipynb       # Data preprocessing, feature engineering, model training
├── Nepal_Rainfall_Enrichment.ipynb  # Rainfall data collection and enrichment pipeline
├── app.py                           # Streamlit application
├── district_list.json               # All 77 Nepal districts with coordinates
├── feature_columns.json             # Feature schema used during training
└── .gitignore
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Scikit-learn | Model training & evaluation |
| OpenMeteo API | Live weather data (rainfall, temperature) |
| Streamlit | Web application interface |
| Jupyter Notebook | Model development & rainfall enrichment |

---

## How to run locally

```bash
git clone https://github.com/manishbhatt1/Nepal-Disaster-Predictor.git
cd Nepal-Disaster-Predictor
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, select any district, and get the current risk prediction.

---

## Data Sources

- Historical flood and landslide records across Nepal (1987–2024)
- Real-time weather: [Open-Meteo API](https://open-meteo.com/) — free, no API key required
- District coordinates: manually compiled for all 77 districts

---

## What's next

- Add earthquake risk as a third prediction category
- District-level risk map visualization
- SMS/alert integration for high-risk periods during monsoon season

---

## Built by

**Manish Bhattarai** — BIT student at Itahari International College, Nepal
[GitHub](https://github.com/manishbhatt1) · [LinkedIn](https://www.linkedin.com/in/manish-bhattarai-ab4248381/)
