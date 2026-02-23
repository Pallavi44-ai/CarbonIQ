# 🌍 CarbonIQ — Global Emission Intelligence Platform

A premium, AI-powered Streamlit dashboard for tracking, forecasting, and analyzing global carbon emissions across 195 countries from 1990–2021.

## 🚀 Quick Start

### 1. Clone / Extract Project
```
carboniq/
├── app.py
├── requirements.txt
├── data/
│   └── final_emissions_dataset.csv
├── models/
│   ├── ml_emission_forecast.pkl
│   └── dl_climate_risk.keras
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── visualization.py
└── ai/
    ├── __init__.py
    └── chatbot.py
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## 🌐 Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file as `app.py`
4. Deploy!

> **Note**: Ensure your `data/` and `models/` folders are included in the repo (or use Git LFS for large model files).

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌐 Global Map | Choropleth world map for any year 1990–2021 |
| 📊 Country Deep Dive | Trend charts, heatmaps, ranking, sector breakdown |
| 🧠 ML Forecast | Scikit-learn powered emission prediction to 2040 |
| 🔥 Climate Risk AI | Keras neural net risk index (temperature + rainfall) |
| ⚖️ Country Comparison | Multi-country side-by-side analytics |
| 🛡️ Policy Advisor | AI-generated, data-driven policy recommendations |
| 🤖 AI Assistant | NLP chatbot with climate expertise |
| 📋 Net Zero Modeling | Pathway modeling to carbon neutrality |

---

## 🛠️ Tech Stack
- **Streamlit** — Web UI
- **Plotly** — Interactive visualizations  
- **Scikit-learn** — ML forecasting
- **TensorFlow/Keras** — DL climate risk model
- **Pandas/NumPy** — Data processing

---

*CarbonIQ — Built for a sustainable future 🌱*
