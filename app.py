# =====================================================================
# TransGlobe Logistics - Business Operations Prediction System
# One-file Streamlit app. Run with:   streamlit run app.py
#
# Pages
#   1. Home               - overview + model summary
#   2. Freight Cost       - predict shipment cost and margin
#   3. Late Delivery      - probability a shipment misses its promise date
#   4. Payment Risk       - probability an invoice is paid late
#   5. Claim Risk         - probability a shipment ends in a claim
#   6. Carrier Ranking    - rank all carriers on cost + reliability
#   7. Anomaly Review     - flag unusual shipments / billing (single or CSV)
# =====================================================================
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TransGlobe Prediction System", page_icon="🚚", layout="wide")

# ---------------------------------------------------------------------
# 1. LOAD THE SAVED MODEL FILES
#    Put this app.py next to (or above) your `model_artifacts_joblib` folder.
# ---------------------------------------------------------------------
OBJECTIVES = ["late_delivery", "freight_cost", "payment_risk", "claim_risk", "anomaly_review"]


def find_model_folder():
    """Look for the folder that holds late_delivery_models.joblib (any depth)."""
    here = Path(__file__).parent
    for hit in here.rglob("late_delivery_models.joblib"):
        return hit.parent
    return None


@st.cache_resource
def load_all():
    folder = find_model_folder()
    if folder is None:
        return None
    store = {}
    for obj in OBJECTIVES:
        scaler_path = folder / f"{obj}_scaler.joblib"
        store[obj] = {
            "models": joblib.load(folder / f"{obj}_models.joblib"),      # dict {name: pipeline}
            "features": joblib.load(folder / f"{obj}_features.joblib"),  # dict with column order
            "scaler": joblib.load(scaler_path) if scaler_path.exists() else None,
        }
    return store


STORE = load_all()
if STORE is None:
    st.error("Could not find the model files. Put the `model_artifacts_joblib` folder "
             "(the one with late_delivery_models.joblib etc.) next to app.py.")
    st.stop()

# ---------------------------------------------------------------------
# 2. DROPDOWN CHOICES (these match the cleaned categories used in training)
# ---------------------------------------------------------------------
COUNTRIES = ["Brazil", "China", "France", "Germany", "India", "Netherlands", "Nigeria",
             "Singapore", "South Africa", "UAE", "United Kingdom", "United States"]
MODES = ["Air", "Sea", "Road", "Rail"]
TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
INDUSTRIES = ["Agriculture", "Automotive", "Electronics", "Energy", "FMCG", "Fashion", "Pharma", "Retail"]
# carrier name -> contract rate per kg (USD) from the carrier table
CARRIERS = {
    "AeroSwift Cargo": 4.47, "BlueWave Maritime": 1.18, "Continental Trucking": 3.78,
    "DeepSea Carriers": 5.03, "EuroLine Transport": 5.17, "IronTrack Rail": 2.98,
    "JetStream Logistics": 1.52, "PortKing Marine": 1.87, "RailCore Express": 1.86,
    "RoadRunner Freight": 3.97, "SilverArrow Air": 3.07, "SkyBridge Air Cargo": 5.98,
    "TransContinental Haulage": 5.22, "oceanlink shipping": 2.34,
}
# a column name prefix in the model  ->  the name of the input column we use
CAT_PREFIXES = {
    "customer_tier_": "customer_tier", "industry_": "industry", "hq_country_": "hq_country",
    "carrier_name_": "carrier_name", "origin_country_": "origin_country",
    "destination_country_": "destination_country", "transport_mode_": "transport_mode",
}

# ---------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------------
def make_frame(raw, objective):
    """Turn plain inputs (one or many rows) into the exact table the model was trained on.
    - start with all zeros using the saved column order
    - copy the numeric values in
    - switch on the matching one-hot (dummy) column for each category
    - scale the continuous columns with the saved scaler
    """
    info = STORE[objective]
    columns = info["features"]["feature_names_in_order"]
    X = pd.DataFrame(0.0, index=range(len(raw)), columns=columns)

    for col in raw.columns:                      # numeric inputs
        if col in X.columns:
            X[col] = raw[col].astype(float).values

    for prefix, name in CAT_PREFIXES.items():    # categorical inputs -> dummy columns
        if name in raw.columns:
            for i, value in enumerate(raw[name].astype(str)):
                dummy = prefix + value
                if dummy in X.columns:
                    X.loc[i, dummy] = 1.0

    scaler = info["scaler"]
    if scaler is not None:                       # scale only what the scaler was fitted on
        cols = list(scaler.feature_names_in_)
        X[cols] = scaler.transform(X[cols])
    return X


def predict_probability(raw, objective, model_name):
    """Probability of the 'yes' class (late / claim / late payment)."""
    model = STORE[objective]["models"][model_name]
    X = make_frame(raw, objective)
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(1)]


def predict_cost(raw, model_name):
    """Predicted total freight cost in USD."""
    model = STORE["freight_cost"]["models"][model_name]
    return model.predict(make_frame(raw, "freight_cost"))


# Isolation Forest never flags anything at its built-in threshold (0 of 2,500 shipments),
# so we flag a shipment when its score is in the top 5% of the scores seen in training.
# Cut-offs below = 95th percentile of the training scores for each detector.
REVIEW_CUTOFF = {"Isolation Forest": -0.1450, "Local Outlier Factor": -0.3424, "One-Class SVM": 0.0}


def anomaly_score(raw, model_name):
    """Higher score = more unusual. Flagged when the score is above the review cut-off."""
    model = STORE["anomaly_review"]["models"][model_name]
    score = -model.decision_function(make_frame(raw, "anomaly_review"))
    return score, score > REVIEW_CUTOFF[model_name]


def risk_level(p):
    if p < 0.35:
        return "Low", "🟢"
    if p < 0.50:
        return "Medium", "🟡"
    return "High", "🔴"


def money(x):
    return f"${x:,.2f}"


def pick_model(objective, default_name, key):
    names = list(STORE[objective]["models"])
    index = names.index(default_name) if default_name in names else 0
    return st.sidebar.selectbox("Model", names, index=index, key=key)


def show_risk(p, what, action_high, action_low):
    level, icon = risk_level(p)
    c1, c2 = st.columns(2)
    c1.metric(f"{what} probability", f"{p:.1%}")
    c2.metric("Risk level", f"{icon} {level}")
    st.progress(float(min(max(p, 0), 1)))
    st.info(action_high if level == "High" else action_low)


def weak_model_warning(what):
    st.warning(f"**Read with care.** On held-out data the {what} models score close to chance "
               "(ROC AUC about 0.5), because this dataset's outcomes were assigned at random. "
               "Use this page to show the workflow, not as a decision tool.")


# --- shared input forms ------------------------------------------------
def customer_inputs(col):
    tier = col.selectbox("Customer tier", TIERS, index=2)
    industry = col.selectbox("Industry", INDUSTRIES)
    hq = col.selectbox("Customer HQ country", COUNTRIES, index=4)
    sla = col.slider("Contract SLA (days)", 1, 14, 5)
    credit = col.number_input("Credit limit (USD)", 0, 1_000_000, 250_000, step=10_000)
    return tier, industry, hq, sla, credit


def route_inputs(col):
    origin = col.selectbox("Origin country", COUNTRIES, index=6)
    dest = col.selectbox("Destination country", COUNTRIES, index=11)
    mode = col.selectbox("Transport mode", MODES)
    weight = col.number_input("Weight (kg)", 1.0, 100_000.0, 200.0, step=10.0)
    month = col.slider("Ship month", 1, 12, 6)
    return origin, dest, mode, weight, month


def shipment_row(carrier, rate, tier, industry, hq, sla, credit, origin, dest, mode, weight, month):
    return pd.DataFrame([{
        "customer_tier": tier, "industry": industry, "hq_country": hq,
        "contract_sla_days": sla, "credit_limit_usd": credit,
        "carrier_name": carrier, "contract_rate_per_kg_usd": rate,
        "origin_country": origin, "destination_country": dest,
        "transport_mode": mode, "weight_kg": weight, "ship_month": month,
    }])


# ---------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ---------------------------------------------------------------------
st.sidebar.title("🚚 TransGlobe")
page = st.sidebar.radio("Go to", [
    "Home", "Freight Cost", "Late Delivery", "Payment Risk",
    "Claim Risk", "Carrier Ranking", "Anomaly Review"])
st.sidebar.markdown("---")

# ---------------------------------------------------------------------
# 5. PAGES
# ---------------------------------------------------------------------
if page == "Home":
    st.title("TransGlobe Logistics: Business Operations Prediction System")
    st.write("Five models that help the operations team cut freight cost, deliver on time, "
             "collect cash faster, choose better carriers and reduce claims.")
    table = pd.DataFrame([
        ["Cut freight cost and protect margin", "Freight Cost", "Regression", "Random Forest Regressor", "Strong"],
        ["Raise on-time delivery and SLA adherence", "Late Delivery", "Classification", "Logistic Regression", "Good"],
        ["Close revenue leakage and speed up cash collection", "Payment Risk + Anomaly Review", "Classification + anomaly", "Logistic Regression / Isolation Forest", "Weak (payment), review tool (anomaly)"],
        ["Rank carriers on cost and reliability", "Carrier Ranking", "Uses the cost and late models", "Combined score", "Follows the two models"],
        ["Reduce claims, damage and loss", "Claim Risk", "Classification", "Logistic Regression", "Weak"],
    ], columns=["Business objective", "Page", "Type", "Default model", "Reliability"])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown("**How to use it:** pick a page in the sidebar, fill in the shipment details, "
                "and read the prediction and the suggested action. You can switch the model in the sidebar.")
    st.caption("Models were trained on a chronological 80/20 split. Payment and claim outcomes in the "
               "source data are random, so those two models score near chance and are shown for workflow demonstration.")

# ---------------------------------------------------------------- freight
elif page == "Freight Cost":
    st.title("💰 Freight Cost and Margin")
    model_name = pick_model("freight_cost", "Random Forest Regressor", "m_cost")
    left, right = st.columns(2)
    tier, industry, hq, sla, credit = customer_inputs(left)
    origin, dest, mode, weight, month = route_inputs(right)
    carrier = right.selectbox("Carrier", list(CARRIERS))
    rate = right.number_input("Contract rate per kg (USD)", 0.0, 50.0, CARRIERS[carrier], step=0.1)
    billed = st.number_input("Amount you plan to bill the customer (USD, optional)", 0.0, 10_000_000.0, 0.0, step=100.0)

    if st.button("Predict freight cost", type="primary"):
        raw = shipment_row(carrier, rate, tier, industry, hq, sla, credit, origin, dest, mode, weight, month)
        cost = max(float(predict_cost(raw, model_name)[0]), 0.0)
        c1, c2 = st.columns(2)
        c1.metric("Predicted freight cost", money(cost))
        c2.metric("Cost per kg", money(cost / weight))
        if billed > 0:
            margin = billed - cost
            st.metric("Expected margin", money(margin), f"{margin / billed:.1%} of billed")
            if margin < 0:
                st.error("Billing is below the predicted cost. Raise the price or switch carrier.")
            elif margin / billed < 0.10:
                st.warning("Margin is thin (under 10%). Check the carrier ranking for a cheaper option.")
            else:
                st.success("Margin looks healthy.")
        else:
            st.info("Enter a billed amount to see the expected margin.")

# ---------------------------------------------------------------- late
elif page == "Late Delivery":
    st.title("⏱️ Late Delivery Risk")
    model_name = pick_model("late_delivery", "Logistic Regression", "m_late")
    left, right = st.columns(2)
    tier, industry, hq, sla, credit = customer_inputs(left)
    origin, dest, mode, weight, month = route_inputs(right)
    carrier = right.selectbox("Carrier", list(CARRIERS))
    rate = right.number_input("Contract rate per kg (USD)", 0.0, 50.0, CARRIERS[carrier], step=0.1)

    if st.button("Check late risk", type="primary"):
        raw = shipment_row(carrier, rate, tier, industry, hq, sla, credit, origin, dest, mode, weight, month)
        p = float(predict_probability(raw, "late_delivery", model_name)[0])
        show_risk(p, "Late delivery",
                  "Action: add buffer time, pick a more reliable carrier, or warn the customer early.",
                  "Action: ship as planned and monitor.")

# ---------------------------------------------------------------- payment
elif page == "Payment Risk":
    st.title("🧾 Payment Risk and Revenue Leakage")
    weak_model_warning("payment-risk")
    model_name = pick_model("payment_risk", "Logistic Regression", "m_pay")
    left, right = st.columns(2)
    tier = left.selectbox("Customer tier", TIERS, index=2)
    industry = left.selectbox("Industry", INDUSTRIES)
    credit = left.number_input("Credit limit (USD)", 0, 1_000_000, 250_000, step=10_000)
    mode = right.selectbox("Transport mode", MODES)
    billed = right.number_input("Billed amount (USD)", 0.0, 10_000_000.0, 2_000.0, step=100.0)
    freight = right.number_input("Total freight cost (USD)", 0.0, 10_000_000.0, 1_500.0, step=100.0)

    if st.button("Check payment risk", type="primary"):
        raw = pd.DataFrame([{
            "customer_tier": tier, "industry": industry, "transport_mode": mode,
            "credit_limit_usd": credit, "billed_amount_usd": billed,
            "total_freight_cost": freight, "margin_usd": billed - freight}])
        p = float(predict_probability(raw, "payment_risk", model_name)[0])
        show_risk(p, "Late payment",
                  "Action: ask for a deposit, shorten payment terms or follow up early.",
                  "Action: standard payment terms.")
        if billed < freight:
            st.error("This invoice bills less than the freight cost: a revenue leak.")

# ---------------------------------------------------------------- claims
elif page == "Claim Risk":
    st.title("📦 Claim, Damage and Loss Risk")
    weak_model_warning("claim-risk")
    model_name = pick_model("claim_risk", "Logistic Regression", "m_claim")
    left, right = st.columns(2)
    tier, industry, hq, sla, credit = customer_inputs(left)
    origin, dest, mode, weight, month = route_inputs(right)
    carrier = right.selectbox("Carrier", list(CARRIERS))
    rate = right.number_input("Contract rate per kg (USD)", 0.0, 50.0, CARRIERS[carrier], step=0.1)

    if st.button("Check claim risk", type="primary"):
        raw = shipment_row(carrier, rate, tier, industry, hq, sla, credit, origin, dest, mode, weight, month)
        p = float(predict_probability(raw, "claim_risk", model_name)[0])
        show_risk(p, "Claim",
                  "Action: improve packaging, add insurance, or choose a carrier with fewer incidents.",
                  "Action: normal handling.")

# ---------------------------------------------------------------- ranking
elif page == "Carrier Ranking":
    st.title("🏆 Carrier Ranking")
    st.write("Scores every carrier for the same shipment using the freight-cost and late-delivery models.")
    cost_model = pick_model("freight_cost", "Random Forest Regressor", "m_rank_cost")
    late_model = pick_model("late_delivery", "Logistic Regression", "m_rank_late")
    left, right = st.columns(2)
    tier, industry, hq, sla, credit = customer_inputs(left)
    origin, dest, mode, weight, month = route_inputs(right)
    w_cost = st.slider("Importance of cost vs reliability", 0, 100, 50,
                       help="100 = only cost matters, 0 = only on-time reliability matters")

    if st.button("Rank carriers", type="primary"):
        rows = pd.concat([shipment_row(c, r, tier, industry, hq, sla, credit, origin, dest, mode, weight, month)
                          for c, r in CARRIERS.items()], ignore_index=True)
        result = pd.DataFrame({
            "Carrier": list(CARRIERS),
            "Predicted cost (USD)": np.maximum(predict_cost(rows, cost_model), 0),
            "Late probability": predict_probability(rows, "late_delivery", late_model),
        })

        def to_01(s):   # 0 = best, 1 = worst
            return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

        result["Score"] = (w_cost / 100) * to_01(result["Predicted cost (USD)"]) \
            + (1 - w_cost / 100) * to_01(result["Late probability"])
        result = result.sort_values("Score").reset_index(drop=True)
        result.insert(0, "Rank", result.index + 1)
        best = result.iloc[0]
        st.success(f"Best carrier for this shipment: **{best['Carrier']}** "
                   f"({money(best['Predicted cost (USD)'])}, {best['Late probability']:.1%} late risk)")
        st.dataframe(result.style.format({"Predicted cost (USD)": "${:,.2f}",
                                          "Late probability": "{:.1%}", "Score": "{:.2f}"}),
                     use_container_width=True, hide_index=True)
        st.caption("Score: 0 is best, 1 is worst. It blends min-max scaled cost and late probability.")
        st.bar_chart(result.set_index("Carrier")["Score"])

# ---------------------------------------------------------------- anomaly
elif page == "Anomaly Review":
    st.title("🔍 Anomaly Review")
    st.write("Flags shipments whose weight, cost, billing and route look unusual. "
             "This is an unsupervised review list, not an accuracy score. A shipment is flagged when it is "
             "more unusual than 95% of the historical shipments.")
    model_name = pick_model("anomaly_review", "Isolation Forest", "m_anom")
    tab1, tab2 = st.tabs(["Single shipment", "Upload CSV"])

    with tab1:
        left, right = st.columns(2)
        weight = left.number_input("Weight (kg)", 1.0, 100_000.0, 200.0, step=10.0)
        freight = left.number_input("Total freight cost (USD)", 0.0, 10_000_000.0, 500.0, step=50.0)
        billed = left.number_input("Total billed amount (USD)", 0.0, 10_000_000.0, 700.0, step=50.0)
        mode = right.selectbox("Transport mode", MODES, key="a_mode")
        origin = right.selectbox("Origin country", COUNTRIES, index=6, key="a_o")
        dest = right.selectbox("Destination country", COUNTRIES, index=11, key="a_d")
        if st.button("Check shipment", type="primary"):
            raw = pd.DataFrame([{
                "weight_kg": weight, "total_freight_cost": freight, "total_billed_amount_usd": billed,
                "cost_per_kg": freight / weight, "margin_usd": billed - freight,
                "transport_mode": mode, "origin_country": origin, "destination_country": dest}])
            score, flag = anomaly_score(raw, model_name)
            c1, c2 = st.columns(2)
            c1.metric("Anomaly score", f"{score[0]:.3f}")
            c2.metric("Result", "🔴 Unusual" if flag[0] else "🟢 Normal")
            st.info("Send this one for manual review." if flag[0] else "Looks like a typical shipment.")

    with tab2:
        st.write("Columns needed: `weight_kg`, `total_freight_cost`, `total_billed_amount_usd`, "
                 "`transport_mode`, `origin_country`, `destination_country`. Use the same country and mode names as the dropdowns (e.g. United Kingdom, not UK).")
        file = st.file_uploader("Upload CSV", type="csv")
        if file is not None:
            data = pd.read_csv(file)
            need = ["weight_kg", "total_freight_cost", "total_billed_amount_usd",
                    "transport_mode", "origin_country", "destination_country"]
            missing = [c for c in need if c not in data.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                raw = data.copy()
                raw["cost_per_kg"] = raw["total_freight_cost"] / raw["weight_kg"].replace(0, np.nan)
                raw["margin_usd"] = raw["total_billed_amount_usd"] - raw["total_freight_cost"]
                score, flag = anomaly_score(raw.fillna(0), model_name)
                data["anomaly_score"] = score
                data["flagged"] = flag
                st.metric("Flagged rows", f"{int(flag.sum())} of {len(data)}")
                st.dataframe(data.sort_values("anomaly_score", ascending=False),
                             use_container_width=True, hide_index=True)
                st.download_button("Download results", data.to_csv(index=False), "anomaly_results.csv")