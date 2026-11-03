from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
st.set_page_config(page_title="Supply Chain Risk Analytics", page_icon="📦", layout="wide")

st.title("📦 Supply Chain Risk Analytics")
st.caption("Big Data Analytics for Supply Chain Risk Assessment and Prediction")
st.info("Demonstration dashboard using synthetic data. Validate all insights against real company data before operational use.")

required = ["shipment_risk_predictions.csv", "supplier_risk_summary.csv", "route_risk_summary.csv", "inventory_risk_summary.csv", "model_metrics.json"]
missing = [f for f in required if not (OUT / f).exists()]
if missing:
    st.error("Pipeline outputs are missing. Run `python run_pipeline.py` in the project folder first.")
    st.stop()

ship = pd.read_csv(OUT / required[0])
sup = pd.read_csv(OUT / required[1])
routes = pd.read_csv(OUT / required[2])
inv = pd.read_csv(OUT / required[3])
metrics = json.loads((OUT / required[4]).read_text())

with st.sidebar:
    st.header("Filters")
    levels = st.multiselect("Risk level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    modes = st.multiselect("Transport mode", sorted(ship["transport_mode"].dropna().unique()),
                           default=sorted(ship["transport_mode"].dropna().unique()))
    st.markdown("---")
    st.write("**Technology:** Python · PySpark-ready · Scikit-learn · Streamlit")

filtered = ship[ship["risk_level"].isin(levels) & ship["transport_mode"].isin(modes)]
total = len(ship)
delay_pct = 100 * ship["is_delayed"].mean() if total else 0
high_risk = int((ship["risk_level"] == "High").sum())
avg_risk = ship["risk_score"].mean() if total else 0

a,b,c,d = st.columns(4)
a.metric("Total Shipments", f"{total:,}")
b.metric("Historical Delay Rate", f"{delay_pct:.1f}%")
c.metric("High-Risk Shipments", f"{high_risk:,}")
d.metric("Average Risk Score", f"{avg_risk:.1f}/100")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Supplier Risk", "Route Risk", "Inventory Risk", "Model & Prediction"])

with tab1:
    left, right = st.columns(2)
    with left:
        risk_counts = filtered["risk_level"].value_counts().rename_axis("Risk Level").reset_index(name="Shipments")
        st.plotly_chart(px.bar(risk_counts, x="Risk Level", y="Shipments", color="Risk Level", title="Shipments by Risk Level"), use_container_width=True)
    with right:
        st.plotly_chart(px.histogram(filtered, x="risk_score", nbins=25, title="Risk Score Distribution"), use_container_width=True)
    st.subheader("Shipment risk records")
    st.dataframe(filtered.sort_values("risk_score", ascending=False).head(100), use_container_width=True)

with tab2:
    st.subheader("Highest-risk suppliers")
    st.plotly_chart(px.bar(sup.head(15), x="supplier_id", y="average_risk_score", color="risk_level",
                           hover_data=["delay_rate", "defect_rate", "supplier_rating"],
                           title="Top 15 suppliers by average risk"), use_container_width=True)
    st.dataframe(sup, use_container_width=True)

with tab3:
    st.subheader("Route performance")
    st.plotly_chart(px.scatter(routes, x="delay_rate", y="average_risk_score", size="shipments", color="average_risk_score",
                               hover_name="route_id", title="Route delay rate vs risk score"), use_container_width=True)
    st.dataframe(routes, use_container_width=True)

with tab4:
    st.subheader("Inventory shortage indicators")
    inv_risk = inv["inventory_risk"].value_counts().rename_axis("Risk Level").reset_index(name="Records")
    st.plotly_chart(px.pie(inv_risk, names="Risk Level", values="Records", title="Inventory risk distribution"), use_container_width=True)
    st.dataframe(inv.head(200), use_container_width=True)

with tab5:
    st.subheader("Model evaluation")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1 Score", f"{metrics['f1_score']:.3f}")
    st.caption(metrics.get("note", ""))
    st.subheader("Try a shipment risk scenario")
    with st.form("predict_form"):
        c1,c2,c3 = st.columns(3)
        supplier_rating = c1.slider("Supplier rating (1–5)", 1.0, 5.0, 3.5, 0.1)
        distance = c2.number_input("Distance (km)", min_value=1, max_value=5000, value=450)
        inventory_level = c3.number_input("Inventory level", min_value=0, max_value=10000, value=150)
        demand = c1.number_input("Demand units", min_value=1, max_value=10000, value=200)
        mode = c2.selectbox("Transport mode", ["Road", "Rail", "Air", "Sea"])
        traffic = c3.selectbox("Traffic level", ["Low", "Medium", "High"])
        weather = c1.selectbox("Weather", ["Clear", "Rain", "Storm"])
        delay_rate = c2.slider("Supplier historical delay rate", 0.0, 1.0, 0.15, 0.01)
        defect_rate = c3.slider("Supplier defect rate", 0.0, 0.5, 0.03, 0.01)
        reliability = c1.slider("Supplier reliability", 0.0, 1.0, 0.85, 0.01)
        expected_days = max(1, int(distance / (700 if mode == "Air" else 300 if mode == "Rail" else 180 if mode == "Sea" else 250)) + 1)
        shipping_cost = distance * 2.5
        month = c2.selectbox("Month", list(range(1,13)), index=5)
        submit = st.form_submit_button("Estimate risk")
    if submit:
        import joblib
        model_path = ROOT / "models" / "shipment_delay_model.joblib"
        model = joblib.load(model_path)
        row = pd.DataFrame([{
            "transport_mode": mode, "traffic_level": traffic, "weather_condition": weather,
            "distance_km": distance, "inventory_level": inventory_level, "demand_units": demand,
            "expected_days": expected_days, "shipping_cost": shipping_cost, "month": month,
            "supplier_rating": supplier_rating, "supplier_delay_rate": delay_rate,
            "supplier_defect_rate": defect_rate, "supplier_reliability": reliability
        }])
        p = float(model.predict_proba(row)[0,1])
        supplier_risk = (1-reliability)*100
        quality_risk = defect_rate*100
        inventory_risk = 100 if inventory_level < demand*0.25 else 60 if inventory_level < demand else 20
        external_risk = ({"Clear":10,"Rain":50,"Storm":95}[weather]*0.6 + {"Low":10,"Medium":50,"High":90}[traffic]*0.4)
        score = round(min(100, max(0, p*100*.30 + quality_risk*.25 + supplier_risk*.20 + inventory_risk*.15 + external_risk*.10)), 1)
        level = "Low" if score <= 30 else "Medium" if score <= 60 else "High"
        st.metric("Estimated delay probability", f"{p*100:.1f}%")
        st.metric("Overall risk score", f"{score}/100")
        st.success(f"Predicted risk level: {level}")
        reasons = []
        if delay_rate >= .2: reasons.append("Supplier has a high historical delay rate.")
        if defect_rate >= .07: reasons.append("Supplier defect rate is elevated.")
        if inventory_level < demand: reasons.append("Inventory is below demand.")
        if weather == "Storm": reasons.append("Storm conditions may disrupt transport.")
        if traffic == "High": reasons.append("High traffic may increase transit time.")
        if not reasons: reasons.append("No extreme risk indicator was triggered by the simple rules.")
        st.write("**Contributing factors and recommendations**")
        for reason in reasons: st.write(f"- {reason}")
        if level == "High":
            st.warning("Suggested action: review supplier alternatives, increase safety stock where justified, and confirm the shipment route/ETA.")
        elif level == "Medium":
            st.info("Suggested action: monitor shipment status and verify inventory coverage.")
        else:
            st.success("Suggested action: continue routine monitoring.")
