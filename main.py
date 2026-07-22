import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from lime.lime_tabular import LimeTabularExplainer

DATA_PATH = "sample_iot_dataset.csv"
MODEL_PATH = "iot_threat_model.joblib"
ATTACK_TYPES = ["DDoS", "Botnet", "Port Scan", "Brute Force", "Malware"]
DEVICE_TYPES = ["Camera", "Smart Thermostat", "Router", "Smart Lock", "Sensor", "Smart TV"]
PROTOCOLS = ["TCP", "UDP", "ICMP"]
FEATURES = ["Packet_Size", "Packet_Rate", "Duration", "Protocol", "Source_Port",
            "Destination_Port", "Bytes", "Flow_Duration", "Device_Type"]
BEHAVIOR_COLS = ["Packet_Rate", "Packet_Size", "Duration", "Bytes", "Flow_Duration"]

# Mean values (packet_size, packet_rate, duration, bytes, flow_duration) per attack type,
# used to sample realistic, distinct traffic signatures.
ATTACK_PROFILES = {
    "DDoS":        (1400, 900, 1, 90000, 2),
    "Botnet":      (300, 400, 20, 30000, 25),
    "Port Scan":   (60, 600, 0.5, 2000, 1),
    "Brute Force": (200, 150, 15, 8000, 18),
    "Malware":     (800, 250, 30, 50000, 35),
}


# ------------------------------- DATASET -------------------------------
def generate_sample_dataset(path=DATA_PATH, n_samples=2000, seed=42):
    """Create a synthetic IoT traffic dataset (if one doesn't already exist)."""
    if os.path.exists(path):
        return pd.read_csv(path)

    rng = np.random.default_rng(seed)
    rows = []
    n_normal = int(n_samples * 0.55)

    for _ in range(n_normal):
        rows.append({
            "Packet_Size": rng.normal(500, 120), "Packet_Rate": rng.normal(30, 10),
            "Duration": rng.normal(5, 2), "Protocol": rng.choice(PROTOCOLS, p=[0.6, 0.3, 0.1]),
            "Source_Port": rng.integers(1024, 65535),
            "Destination_Port": rng.choice([80, 443, 22, 21, 8080]),
            "Bytes": rng.normal(4000, 1000), "Flow_Duration": rng.normal(10, 3),
            "Device_Type": rng.choice(DEVICE_TYPES), "Attack_Type": "None", "Label": 0,
        })

    per_attack = (n_samples - n_normal) // len(ATTACK_TYPES)
    for attack, (ps, pr, dur, byt, fd) in ATTACK_PROFILES.items():
        for _ in range(per_attack):
            rows.append({
                "Packet_Size": rng.normal(ps, ps * 0.15), "Packet_Rate": rng.normal(pr, pr * 0.2),
                "Duration": rng.normal(dur, max(dur * 0.3, 0.1)),
                "Protocol": rng.choice(PROTOCOLS, p=[0.5, 0.4, 0.1]),
                "Source_Port": rng.integers(1024, 65535),
                "Destination_Port": rng.choice([80, 443, 22, 3389, 23]),
                "Bytes": rng.normal(byt, byt * 0.2), "Flow_Duration": rng.normal(fd, max(fd * 0.3, 0.1)),
                "Device_Type": rng.choice(DEVICE_TYPES), "Attack_Type": attack, "Label": 1,
            })

    df = pd.DataFrame(rows)
    for col in BEHAVIOR_COLS:
        df[col] = df[col].clip(lower=1)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


# ---------------------------- PREPROCESSING -----------------------------
def preprocess_data(df):
    """Drop duplicates, fill missing values, encode categoricals, split train/test."""
    df = df.drop_duplicates().copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in ["Protocol", "Device_Type"]:
        df[col] = LabelEncoder().fit_transform(df[col])

    X, y = df[FEATURES].copy(), df["Label"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    return df, X, y, X_train, X_test, y_train, y_test


# ------------------------------- MODEL -----------------------------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred), "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0), "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    }
    return metrics, confusion_matrix(y_test, y_pred)


# --------------------------- ADAPTIVE RISK SCORE --------------------------
def compute_risk_scores(df, attack_probs):
    """Weighted 0-100 risk score: attack probability + scaled traffic behavior features."""
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(df[BEHAVIOR_COLS]), columns=BEHAVIOR_COLS, index=df.index)
    weights = {"Packet_Rate": 0.20, "Bytes": 0.15, "Packet_Size": 0.08, "Flow_Duration": 0.07, "Duration": 0.05}
    risk = 0.45 * attack_probs * 100
    for col, w in weights.items():
        risk += w * scaled[col] * 100
    return risk.clip(0, 100).round(2)


def classify_risk(score):
    if score >= 80: return "Critical"
    if score >= 55: return "High"
    if score >= 30: return "Medium"
    return "Low"


# ----------------------------- RECOMMENDATIONS -----------------------------
BASE_ACTIONS = {
    "DDoS": ["Enable Firewall Rate-Limiting", "Block Source IP", "Increase Logging"],
    "Botnet": ["Disconnect Device", "Block Source IP", "Change Credentials"],
    "Port Scan": ["Enable Firewall", "Monitor Device", "Increase Logging"],
    "Brute Force": ["Change Credentials", "Block Source IP", "Enable Firewall"],
    "Malware": ["Disconnect Device", "Change Credentials", "Increase Logging"],
    "None": ["Monitor Device"],
}


def get_recommendations(attack_type, risk_level):
    actions = list(BASE_ACTIONS.get(attack_type, ["Monitor Device"]))
    if risk_level == "Critical" and "Disconnect Device" not in actions:
        actions.insert(0, "Disconnect Device")
    if risk_level in ("Critical", "High") and "Increase Logging" not in actions:
        actions.append("Increase Logging")
    if risk_level == "Low":
        actions = ["Monitor Device"]
    seen = set()
    return [a for a in actions if not (a in seen or seen.add(a))]


# ---------------------------- NETWORK HEALTH -------------------------------
def compute_network_health(results_df):
    total = len(results_df)
    if total == 0:
        return 100.0, "Excellent"
    attack_ratio = (results_df["Prediction"] == "Attack").mean()
    penalty = 0.5 * attack_ratio * 100 + 0.3 * results_df["Risk_Score"].mean() + 0.2 * results_df["Attack_Probability (%)"].mean()
    health = round(max(0, 100 - penalty), 2)
    label = "Excellent" if health >= 85 else "Good" if health >= 65 else "Warning" if health >= 40 else "Critical"
    return health, label


# -------------------------------- LIME --------------------------------------
def build_lime_explainer(X_train):
    return LimeTabularExplainer(X_train.values, feature_names=FEATURES, class_names=["Normal", "Attack"],
                                 mode="classification", discretize_continuous=True)


def explain_instance(explainer, model, instance_row, num_features=6):
    exp = explainer.explain_instance(instance_row.values.astype(float), model.predict_proba, num_features=num_features)
    return exp.as_list()


# ------------------------------ DASHBOARD ------------------------------------
def main():
    st.set_page_config(page_title="Adaptive IoT Threat Prioritization", layout="wide")
    st.title("🛡️ Adaptive IoT Threat Prioritization and Risk Assessment")
    st.caption("A Machine Learning based SOC decision-support dashboard for IoT networks")

    st.sidebar.header("⚙️ Controls")
    uploaded_file = st.sidebar.file_uploader("Upload IoT traffic CSV (optional)", type=["csv"])
    run_button = st.sidebar.button("🚀 Run Threat Detection")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom dataset loaded.")
    else:
        raw_df = generate_sample_dataset()
        st.sidebar.info("Using auto-generated sample_iot_dataset.csv")

    st.subheader("📄 Dataset Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    for col, default in [("Attack_Type", "None"), ("Label", 0)]:
        if col not in raw_df.columns:
            raw_df[col] = default

    df, X, y, X_train, X_test, y_train, y_test = preprocess_data(raw_df)

    if run_button or "model" not in st.session_state:
        model = train_model(X_train, y_train)
        st.session_state["model"], st.session_state["X_train"] = model, X_train
    else:
        model, X_train = st.session_state["model"], st.session_state["X_train"]

    st.subheader("📊 Model Performance")
    metrics, cm = evaluate_model(model, X_test, y_test)
    cols = st.columns(4)
    for c, (name, val) in zip(cols, metrics.items()):
        c.metric(name, f"{val*100:.2f}%")

    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Attack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Attack"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col_fi:
        st.markdown("**Feature Importance**")
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ax.barh(imp.index, imp.values, color="#4C72B0")
        st.pyplot(fig)

    st.subheader("🚨 Threat Detection & Adaptive Risk Prioritization")
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    risk = compute_risk_scores(df, probs)

    results = df.copy()
    results["Prediction"] = np.where(preds == 1, "Attack", "Normal")
    results["Attack_Probability (%)"] = (probs * 100).round(2)
    results["Confidence (%)"] = (np.max(model.predict_proba(X), axis=1) * 100).round(2)
    results["Risk_Score"] = risk
    results["Risk_Level"] = risk.apply(classify_risk)
    results["Recommended_Actions"] = [", ".join(get_recommendations(a, l))
                                       for a, l in zip(results["Attack_Type"], results["Risk_Level"])]

    ranked = results.sort_values("Risk_Score", ascending=False)
    cols_show = ["Device_Type", "Attack_Type", "Prediction", "Attack_Probability (%)",
                 "Confidence (%)", "Risk_Score", "Risk_Level", "Recommended_Actions"]
    st.markdown("**Ranked Threat Table (highest risk first)**")
    st.dataframe(ranked[cols_show].head(50), use_container_width=True)

    buf = io.StringIO()
    ranked[cols_show].to_csv(buf, index=False)
    st.download_button("⬇️ Download Prediction Results (CSV)", buf.getvalue(), "threat_predictions.csv", "text/csv")

    st.subheader("💚 Network Health Score")
    health, label = compute_network_health(results)
    color = {"Excellent": "green", "Good": "blue", "Warning": "orange", "Critical": "red"}[label]
    st.markdown(f"### Health Score: **{health}/100** — :{color}[{label}]")
    st.progress(int(health))

    st.subheader("🔍 Explainable AI (LIME)")
    st.caption("Select a record to see why the model made its decision.")
    record_index = st.selectbox("Choose a record (by dataframe index) to explain:", list(ranked.index[:20]))

    if st.button("Explain This Prediction"):
        explainer = build_lime_explainer(X_train)
        explanation = explain_instance(explainer, model, X.loc[record_index])
        exp_df = pd.DataFrame(explanation, columns=["Feature Condition", "Weight"])
        exp_df["Effect"] = np.where(exp_df["Weight"] > 0, "➕ Pushes toward Attack", "➖ Pushes toward Normal")
        st.dataframe(exp_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ["#d62728" if w > 0 else "#1f77b4" for _, w in explanation]
        ax.barh([f[0] for f in explanation], [f[1] for f in explanation], color=colors)
        ax.set_xlabel("Contribution to Prediction")
        st.pyplot(fig)
        st.info("🟥 Red bars push toward **Attack**. 🟦 Blue bars push toward **Normal**.")

    st.markdown("---")
    st.caption("Adaptive IoT Threat Prioritization | Random Forest + Weighted Risk Scoring + LIME | Streamlit")


if __name__ == "__main__":
    main()
