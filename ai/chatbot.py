import pandas as pd
import numpy as np
import random

RESPONSE_BANK = {
    "highest emissions": [
        "Based on our dataset, {top_country} consistently shows the highest total emissions across sectors, reaching {top_val:.1f} tCO₂e in the latest year.",
        "The data reveals {top_country} leads global emissions at {top_val:.1f} tCO₂e — driven largely by industrial and energy sectors."
    ],
    "reduce transport": [
        "To reduce transport emissions: (1) Electrify vehicle fleets — EVs cut per-km CO₂ by ~70%. (2) Invest in public transit and rail. (3) Implement carbon pricing on aviation fuel. (4) Promote active mobility like cycling infrastructure. (5) Optimize freight logistics using AI routing.",
        "Transport decarbonization starts with modal shift — rail emits ~80% less CO₂/km than aviation. Combined with EV mandates, urban density policies, and last-mile electrification, net-zero transport is achievable by 2050."
    ],
    "paris agreement": [
        "The 2015 Paris Agreement set a global goal to limit warming to 1.5–2°C above pre-industrial levels. Countries submit Nationally Determined Contributions (NDCs) detailing emission reduction plans. Current pledges are still insufficient to meet 1.5°C — a ~50% global cut by 2030 is required.",
        "Under Paris, developed nations pledged $100B/year in climate finance to developing countries. Progress has been mixed — the IPCC warns emissions must peak before 2025 to stay on track."
    ],
    "carbon neutral": [
        "Carbon neutrality means net-zero CO₂ emissions — achieved by balancing remaining emissions with carbon removal (forests, DAC, CCS). Steps: (1) Energy audit & efficiency measures (2) Switch to renewables (3) Electrify heating & transport (4) Offset unavoidable emissions via verified credits.",
        "Industries can reach carbon neutrality via: process electrification, green hydrogen for high-heat processes, circular economy (reducing material waste), and science-based targets (SBTi) setting verified reduction pathways."
    ],
    "effects climate change": [
        "Documented effects include: sea-level rise threatening 1B coastal residents, intensified droughts cutting crop yields 10-25%, 1°C warming already pushing coral bleaching, and extreme weather events increasing 5x since 1970. Economic losses from climate disasters hit $313B in 2022.",
        "Climate change creates feedback loops: Arctic ice melt reduces albedo → more warming; permafrost thaw releases methane → accelerating change. By 2100, unchecked warming could reduce global GDP by 10-23%."
    ],
    "renewable energy": [
        "Solar and wind now provide the cheapest electricity ever recorded — solar costs fell 90% since 2010. In 2023, renewables added 295 GW globally. The IEA projects renewables covering 90% of new power capacity through 2030.",
        "Key renewable strategies: utility-scale solar (best for sunbelt countries), offshore wind (ideal for coastal nations), geothermal (excellent baseload), and pumped hydro storage to manage intermittency."
    ],
    "india emissions": [
        "India is the 3rd largest emitter globally, with emissions dominated by coal-fired power (~45%). However, per-capita emissions remain ~2.5 tCO₂ — much lower than developed nations. India's NDC targets 45% carbon intensity reduction by 2030 and 500 GW renewable capacity.",
    ],
    "china emissions": [
        "China accounts for ~30% of global emissions, primarily from coal (57% of energy mix), cement production, and steel manufacturing. China has pledged carbon neutrality by 2060 and peak emissions before 2030. It leads globally in solar, wind, and EV deployment.",
    ],
    "default": [
        "Great question! Based on our global emissions dataset spanning 1990–2021, I can tell you that emissions have risen ~60% since the early 1990s, with the largest contributions from energy and transport sectors. Would you like specific country or sector analysis?",
        "This is a critical area of concern. The data shows emissions remain on a dangerous trajectory — global action across energy transition, industrial decarbonization, and land use change is essential. What specific aspect would you like to explore?",
        "Our dataset captures emissions across 195 countries from 1990–2021. Key insight: while absolute emissions have grown, carbon intensity (emissions per GDP) has actually fallen in many developed economies — showing decoupling is possible.",
    ]
}

def match_intent(query):
    q = query.lower()
    
    if any(w in q for w in ["highest", "most", "top", "biggest", "largest"]):
        return "highest emissions"
    elif any(w in q for w in ["transport", "vehicle", "car", "aviation", "fleet"]):
        return "reduce transport"
    elif any(w in q for w in ["paris", "agreement", "cop"]):
        return "paris agreement"
    elif any(w in q for w in ["carbon neutral", "net zero", "neutrality", "offset"]):
        return "carbon neutral"
    elif any(w in q for w in ["effect", "impact", "consequence", "warming"]):
        return "effects climate change"
    elif any(w in q for w in ["renewable", "solar", "wind", "clean energy"]):
        return "renewable energy"
    elif "india" in q:
        return "india emissions"
    elif "china" in q:
        return "china emissions"
    else:
        return "default"

def ai_assistant(query, df):
    intent = match_intent(query)
    
    # Get contextual data
    latest_year = df["Year"].max()
    ydf = df[df["Year"] == latest_year].groupby("Country")["Emissions"].sum()
    top_country = ydf.idxmax()
    top_val = ydf.max()
    
    templates = RESPONSE_BANK.get(intent, RESPONSE_BANK["default"])
    response = random.choice(templates)
    
    try:
        response = response.format(top_country=top_country, top_val=top_val)
    except:
        pass
    
    return response

def get_policy_recommendations(country, emissions, trend, df):
    """Generate detailed policy recommendations based on country data."""
    
    recommendations = []
    
    # Base on emission level
    if emissions > 1000:
        recommendations.extend([
            {"priority": "🔴 Critical", "action": "Emergency carbon tax implementation", 
             "impact": "15-25% reduction in 5 years", "category": "Policy"},
            {"priority": "🔴 Critical", "action": "Coal phase-out mandate with 2035 deadline",
             "impact": "40% energy sector reduction", "category": "Energy"},
            {"priority": "🔴 Critical", "action": "Heavy industry carbon capture mandate",
             "impact": "20% industrial emissions cut", "category": "Industry"},
        ])
    elif emissions > 300:
        recommendations.extend([
            {"priority": "🟠 High", "action": "Carbon pricing scheme (ETS or tax)",
             "impact": "10-18% reduction in 7 years", "category": "Policy"},
            {"priority": "🟠 High", "action": "Renewable energy subsidy acceleration",
             "impact": "30% clean energy share by 2030", "category": "Energy"},
        ])
    else:
        recommendations.extend([
            {"priority": "🟡 Medium", "action": "Green building codes & retrofit programs",
             "impact": "8-12% building sector reduction", "category": "Buildings"},
            {"priority": "🟡 Medium", "action": "Sustainable agriculture incentives",
             "impact": "5-10% land-use emissions cut", "category": "Agriculture"},
        ])
    
    # Trend-based
    if trend == "increasing":
        recommendations.append({
            "priority": "🔴 Urgent", "action": "Immediate emission monitoring & reporting mandate",
            "impact": "Enables targeted intervention", "category": "Governance"
        })
    
    # Universal
    recommendations.extend([
        {"priority": "🟢 All", "action": "National EV transition plan with incentives",
         "impact": "60-70% transport emission cut", "category": "Transport"},
        {"priority": "🟢 All", "action": "Afforestation & ecosystem restoration",
         "impact": "Natural carbon sink enhancement", "category": "Nature"},
    ])
    
    return recommendations
