import pandas as pd
import numpy as np

def clean_dataset(df):
    df = df.dropna(subset=["Country", "Year", "Emissions"])
    df["Year"] = df["Year"].astype(int)
    df["Emissions"] = pd.to_numeric(df["Emissions"], errors="coerce").fillna(0)
    df["Sector"] = df["Sector"].fillna("Unknown")
    return df

def compute_kpis(df):
    latest_year = df["Year"].max()
    prev_year = latest_year - 1

    latest = df[df["Year"] == latest_year]["Emissions"].sum()
    prev = df[df["Year"] == prev_year]["Emissions"].sum()
    yoy_change = ((latest - prev) / prev * 100) if prev != 0 else 0

    top_country = df[df["Year"] == latest_year].groupby("Country")["Emissions"].sum().idxmax()
    top_emission = df[df["Year"] == latest_year].groupby("Country")["Emissions"].sum().max()

    total_countries = df["Country"].nunique()
    avg_emission = df[df["Year"] == latest_year]["Emissions"].mean()

    return {
        "total_latest": latest,
        "yoy_change": yoy_change,
        "top_country": top_country,
        "top_emission": top_emission,
        "total_countries": total_countries,
        "avg_emission": avg_emission,
        "latest_year": latest_year
    }

def get_country_stats(df, country):
    cdf = df[df["Country"] == country].copy()
    if cdf.empty:
        return {}
    
    latest_year = cdf["Year"].max()
    latest_emissions = cdf[cdf["Year"] == latest_year]["Emissions"].sum()
    
    # Trend: linear regression
    by_year = cdf.groupby("Year")["Emissions"].sum().reset_index()
    if len(by_year) > 1:
        x = by_year["Year"].values
        y = by_year["Emissions"].values
        slope = np.polyfit(x, y, 1)[0]
        trend = "increasing" if slope > 0 else "decreasing"
    else:
        trend = "stable"
        slope = 0
    
    peak_year = by_year.loc[by_year["Emissions"].idxmax(), "Year"]
    peak_val = by_year["Emissions"].max()
    
    return {
        "latest_year": latest_year,
        "latest_emissions": latest_emissions,
        "trend": trend,
        "slope": slope,
        "peak_year": peak_year,
        "peak_val": peak_val
    }

def compute_risk_level(emission_val):
    if emission_val > 1000:
        return "🔴 Critical", "critical"
    elif emission_val > 500:
        return "🟠 High", "high"
    elif emission_val > 200:
        return "🟡 Moderate", "moderate"
    else:
        return "🟢 Low", "low"
