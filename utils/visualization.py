import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

THEME = {
    "bg": "#050d1a",
    "card": "#0a1628",
    "accent1": "#00e5ff",
    "accent2": "#39ff14",
    "accent3": "#ff6b35",
    "accent4": "#bf5af2",
    "text": "#e2e8f0",
    "muted": "#64748b",
    "grid": "#1e293b",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Syne', sans-serif", color=THEME["text"]),
        xaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"]),
        yaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=THEME["grid"]),
        colorway=[THEME["accent1"], THEME["accent2"], THEME["accent3"], THEME["accent4"], "#f59e0b", "#ec4899"],
        margin=dict(t=40, b=40, l=40, r=20),
    )
)

def emission_trend_chart(df, country, sector):
    cdf = df[(df["Country"] == country) & (df["Sector"] == sector)].sort_values("Year")
    
    if cdf.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available", **PLOTLY_TEMPLATE["layout"])
        return fig

    # Add trend line
    x = cdf["Year"].values
    y = cdf["Emissions"].values
    z = np.polyfit(x, y, 1)
    trend_y = np.poly1d(z)(x)

    fig = go.Figure()
    
    # Area fill
    fig.add_trace(go.Scatter(
        x=cdf["Year"], y=cdf["Emissions"],
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.08)",
        line=dict(color=THEME["accent1"], width=2.5),
        mode="lines+markers",
        marker=dict(size=6, color=THEME["accent1"], symbol="circle"),
        name=f"{sector} Emissions",
        hovertemplate="<b>%{x}</b><br>Emissions: %{y:.2f} tCO₂e<extra></extra>"
    ))
    
    # Trend line
    fig.add_trace(go.Scatter(
        x=cdf["Year"], y=trend_y,
        line=dict(color=THEME["accent3"], width=1.5, dash="dot"),
        name="Trend",
        hovertemplate="Trend: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=f"<b>{country}</b> — {sector} Emissions Over Time", font=dict(size=16, color=THEME["accent1"])),
        xaxis_title="Year",
        yaxis_title="tCO₂e",
        **PLOTLY_TEMPLATE["layout"]
    )
    return fig


def global_choropleth(df, year):
    ydf = df[df["Year"] == year].groupby("Country")["Emissions"].sum().reset_index()
    
    fig = px.choropleth(
        ydf,
        locations="Country",
        locationmode="country names",
        color="Emissions",
        color_continuous_scale=[[0, "#0a1628"], [0.3, "#0e4d6e"], [0.6, "#ff6b35"], [1, "#ff0040"]],
        title=f"<b>Global Emissions Map — {year}</b>",
        labels={"Emissions": "tCO₂e"},
    )
    fig.update_layout(
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#1e293b",
            showland=True,
            landcolor="#0a1628",
            showocean=True,
            oceancolor="#050d1a",
            showlakes=True,
            lakecolor="#050d1a",
        ),
        coloraxis_colorbar=dict(title="tCO₂e", tickfont=dict(color=THEME["text"])),
        title_font=dict(size=16, color=THEME["accent1"]),
        **PLOTLY_TEMPLATE["layout"]
    )
    return fig


def top_emitters_bar(df, year, n=15):
    ydf = df[df["Year"] == year].groupby("Country")["Emissions"].sum().reset_index()
    top = ydf.nlargest(n, "Emissions")
    
    colors = [f"rgba(0,229,255,{0.4 + 0.6 * i/n})" for i in range(n)]
    
    fig = go.Figure(go.Bar(
        x=top["Emissions"],
        y=top["Country"],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=THEME["accent1"], width=0.5)
        ),
        hovertemplate="<b>%{y}</b><br>%{x:.1f} tCO₂e<extra></extra>",
        text=top["Emissions"].round(1),
        textposition="outside",
        textfont=dict(color=THEME["text"], size=11)
    ))
    
    fig.update_layout(
        title=dict(text=f"<b>Top {n} Emitters — {year}</b>", font=dict(size=15, color=THEME["accent1"])),
        xaxis_title="Emissions (tCO₂e)",
        **PLOTLY_TEMPLATE["layout"]
    )
    return fig


def sector_pie(df, country):
    latest_year = df["Year"].max()
    cdf = df[(df["Country"] == country) & (df["Year"] == latest_year)]
    sdf = cdf.groupby("Sector")["Emissions"].sum().reset_index()
    
    if sdf.empty:
        return go.Figure()
    
    fig = go.Figure(go.Pie(
        labels=sdf["Sector"],
        values=sdf["Emissions"],
        hole=0.55,
        marker=dict(
            colors=[THEME["accent1"], THEME["accent2"], THEME["accent3"], THEME["accent4"]],
            line=dict(color=THEME["bg"], width=2)
        ),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:.2f} tCO₂e (%{percent})<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=f"<b>{country}</b> — Emission by Sector", font=dict(size=15, color=THEME["accent2"])),
        annotations=[dict(text=country[:3], font=dict(size=20, color=THEME["accent1"]), showarrow=False)],
        **PLOTLY_TEMPLATE["layout"]
    )
    return fig


def global_trend_line(df):
    gdf = df.groupby("Year")["Emissions"].sum().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gdf["Year"], y=gdf["Emissions"],
        fill="tozeroy",
        fillcolor="rgba(57,255,20,0.07)",
        line=dict(color=THEME["accent2"], width=2.5),
        mode="lines",
        name="Global Total",
        hovertemplate="<b>%{x}</b><br>Total: %{y:,.0f} tCO₂e<extra></extra>"
    ))

    # Paris Agreement reference line (2015)
    fig.add_vline(x=2015, line=dict(color=THEME["accent3"], dash="dash", width=1.5),
                  annotation_text="Paris Agreement", annotation_font_color=THEME["accent3"])

    fig.update_layout(
        title=dict(text="<b>Global Emission Trend (All Countries)</b>", font=dict(size=15, color=THEME["accent2"])),
        xaxis_title="Year", yaxis_title="Total tCO₂e",
        **PLOTLY_TEMPLATE["layout"]
    )
    return fig


def emission_heatmap(df, country):
    cdf = df[df["Country"] == country].copy()
    if cdf.empty:
        return go.Figure()
    
    pivot = cdf.pivot_table(index="Sector", columns="Year", values="Emissions", aggfunc="sum")
    
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#050d1a"], [0.5, "#0e4d6e"], [1, "#00e5ff"]],
        hovertemplate="Year: %{x}<br>Sector: %{y}<br>%{z:.2f} tCO₂e<extra></extra>",
        showscale=True,
    ))
    
    fig.update_layout(
        title=dict(text=f"<b>{country}</b> — Sector × Year Heatmap", font=dict(size=15, color=THEME["accent1"])),
        **PLOTLY_TEMPLATE["layout"]
    )
    return fig


def scatter_comparison(df, year):
    ydf = df[df["Year"] == year].groupby("Country")["Emissions"].sum().reset_index()
    ydf["Log_Emissions"] = np.log1p(ydf["Emissions"])
    
    # Add continent (simplified mapping)
    continent_map = {
        "United States": "Americas", "China": "Asia", "India": "Asia",
        "Germany": "Europe", "France": "Europe", "Brazil": "Americas",
        "Russia": "Europe", "Japan": "Asia", "United Kingdom": "Europe",
        "Canada": "Americas", "Australia": "Oceania", "South Africa": "Africa",
        "Nigeria": "Africa", "Egypt": "Africa", "Mexico": "Americas"
    }
    ydf["Continent"] = ydf["Country"].map(continent_map).fillna("Other")
    
    fig = px.scatter(
        ydf, x="Emissions", y="Log_Emissions",
        size="Emissions", color="Continent",
        hover_name="Country",
        color_discrete_map={
            "Americas": THEME["accent3"], "Asia": THEME["accent1"],
            "Europe": THEME["accent4"], "Africa": THEME["accent2"],
            "Oceania": "#f59e0b", "Other": THEME["muted"]
        },
        labels={"Emissions": "tCO₂e", "Log_Emissions": "Log Scale"},
        title=f"<b>Emissions Scatter — {year}</b>",
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>%{x:.2f} tCO₂e<extra></extra>",
        marker=dict(opacity=0.8, line=dict(width=0.5, color="white"))
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig


def forecast_chart(country, year_from, year_to, ml_model, df):
    cdf = df[df["Country"] == country].groupby("Year")["Emissions"].sum().reset_index()
    
    years_future = list(range(year_from, year_to + 1))
    preds = [ml_model.predict([[y]])[0] for y in years_future]
    
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=cdf["Year"], y=cdf["Emissions"],
        line=dict(color=THEME["accent1"], width=2),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.07)",
        name="Historical",
        hovertemplate="<b>%{x}</b>: %{y:.2f} tCO₂e<extra></extra>"
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=years_future, y=preds,
        line=dict(color=THEME["accent3"], width=2.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.07)",
        name="ML Forecast",
        hovertemplate="<b>Forecast %{x}</b>: %{y:.2f}<extra></extra>"
    ))
    
    # Boundary
    fig.add_vline(x=year_from, line=dict(color=THEME["muted"], dash="dash", width=1))
    fig.add_annotation(x=year_from, y=max(cdf["Emissions"].max(), max(preds)),
                       text="Forecast →", showarrow=False,
                       font=dict(color=THEME["accent3"], size=12))
    
    fig.update_layout(
        title=dict(text=f"<b>Emission Forecast: {country} ({year_from}–{year_to})</b>",
                   font=dict(size=15, color=THEME["accent1"])),
        xaxis_title="Year", yaxis_title="tCO₂e",
        **PLOTLY_TEMPLATE["layout"]
    )
    return fig, preds, years_future


def multi_country_comparison(df, countries, sector):
    fig = go.Figure()
    colors = [THEME["accent1"], THEME["accent2"], THEME["accent3"], THEME["accent4"], "#f59e0b"]
    
    for i, country in enumerate(countries):
        cdf = df[(df["Country"] == country) & (df["Sector"] == sector)].sort_values("Year")
        if not cdf.empty:
            fig.add_trace(go.Scatter(
                x=cdf["Year"], y=cdf["Emissions"],
                name=country,
                line=dict(width=2.5, color=colors[i % len(colors)]),
                mode="lines+markers",
                marker=dict(size=5),
                hovertemplate=f"<b>{country}</b> %{{x}}: %{{y:.2f}} tCO₂e<extra></extra>"
            ))
    
    fig.update_layout(
        title=dict(text=f"<b>Multi-Country Comparison — {sector}</b>", font=dict(size=15, color=THEME["accent1"])),
        xaxis_title="Year", yaxis_title="tCO₂e",
        **PLOTLY_TEMPLATE["layout"]
    )
    return fig
