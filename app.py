"""
Streamlit frontend for the Land Acquisition Delay Prediction project.

This app is a UI layer only. All ML logic (preprocessing, scaling, the
trained model) is loaded from artifacts produced by train_and_export.py,
which reproduces the original Colab notebook's steps exactly (see that
file's docstring for the cell-by-cell mapping). Nothing here retrains,
replaces, or alters the model, the scaler, or the feature engineering.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "model"

st.set_page_config(
    page_title="Land Acquisition Delay Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --brand-primary: #1f6f54;
    --brand-primary-dark: #16513d;
    --brand-danger: #b3261e;
    --brand-bg-card: #ffffff;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

.hero {
    background: linear-gradient(135deg, #1f6f54 0%, #16513d 100%);
    padding: 2rem 2.25rem;
    border-radius: 14px;
    color: #ffffff;
    margin-bottom: 1.5rem;
}
.hero h1 { margin: 0 0 0.35rem 0; font-size: 1.9rem; }
.hero p { margin: 0; opacity: 0.92; font-size: 1.02rem; }

.metric-card {
    background: var(--brand-bg-card);
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric-card .label { font-size: 0.82rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
.metric-card .value { font-size: 1.7rem; font-weight: 700; color: #111827; margin-top: 0.15rem; }

.result-card {
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-top: 1rem;
    color: white;
    text-align: center;
}
.result-delayed { background: linear-gradient(135deg, #b3261e 0%, #7f1d1a 100%); }
.result-ontime { background: linear-gradient(135deg, #1f6f54 0%, #16513d 100%); }
.result-card .headline { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; }
.result-card .subline { font-size: 1rem; opacity: 0.9; }

.section-note {
    background: #f3f4f6;
    border-left: 4px solid var(--brand-primary);
    padding: 0.75rem 1rem;
    border-radius: 6px;
    font-size: 0.9rem;
    color: #374151;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    required = [
        "final_model.pkl", "scaler.pkl", "feature_columns.json",
        "numeric_features.json", "categorical_options.json",
        "numeric_stats.json", "metrics.json",
    ]
    missing = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): " + ", ".join(missing) + ". "
            "Run `python train_and_export.py --csv <path-to-csv>` first."
        )

    with open(MODEL_DIR / "final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(MODEL_DIR / "numeric_features.json") as f:
        numeric_features = json.load(f)
    with open(MODEL_DIR / "categorical_options.json") as f:
        categorical_options = json.load(f)
    with open(MODEL_DIR / "numeric_stats.json") as f:
        numeric_stats = json.load(f)
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)

    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_options": categorical_options,
        "numeric_stats": numeric_stats,
        "metrics": metrics,
    }


def predict_delay(raw_inputs: dict, artifacts: dict):
    """
    Mirrors predict_with_user_input() from the original notebook (cell 42),
    with input() calls replaced by values collected from the Streamlit form.
    Same steps: zero-filled row shaped like X_train -> populate numeric raw
    values -> set one-hot dummy flags -> scale numeric_features with the
    fitted scaler -> reorder columns -> model.predict / predict_proba.
    """
    feature_columns = artifacts["feature_columns"]
    numeric_features = artifacts["numeric_features"]
    categorical_options = artifacts["categorical_options"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]

    row = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    for feature in numeric_features:
        row.loc[0, feature] = float(raw_inputs[feature])

    for col, meta in categorical_options.items():
        selected = raw_inputs[col]
        if selected != meta["base_category"]:
            dummy_col = f"{col}_{selected}"
            if dummy_col in row.columns:
                row.loc[0, dummy_col] = 1.0

    row[numeric_features] = scaler.transform(row[numeric_features])
    row = row[feature_columns]

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0]
    return int(prediction), probability


def render_home():
    st.markdown(
        """
        <div class="hero">
            <h1>🏗️ Land Acquisition Delay Predictor</h1>
            <p>Predicts whether a land acquisition project is likely to be delayed,
            based on ownership complexity, legal/document status, and project scope.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("What this does")
        st.write(
            "This tool wraps a trained machine learning model that classifies a land "
            "acquisition project as **Delayed** or **Not Delayed**, using project "
            "attributes such as land area, number of owners/parcels, pending legal "
            "cases, missing documents, objections, government clearances, and planned "
            "timelines."
        )
        st.write(
            "The model is a **Logistic Regression** classifier trained on a dataset "
            "of 5,000 historical land acquisition projects. All preprocessing "
            "(missing-value imputation, feature scaling, one-hot encoding of "
            "categorical fields) and the trained model itself are taken directly from "
            "the original ML notebook — this app does not retrain or alter them."
        )
        st.markdown(
            '<div class="section-note">Go to <b>🔮 Prediction</b> in the sidebar to '
            "score a new project, or <b>📊 Model Performance</b> to see the metrics "
            "produced during model development.</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.subheader("At a glance")
        st.markdown(
            """
            <div class="metric-card"><div class="label">Task</div><div class="value" style="font-size:1.1rem;">Binary Classification</div></div>
            <br>
            <div class="metric-card"><div class="label">Algorithm</div><div class="value" style="font-size:1.1rem;">Logistic Regression</div></div>
            <br>
            <div class="metric-card"><div class="label">Training Records</div><div class="value" style="font-size:1.1rem;">5,000 projects</div></div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction(artifacts: dict):
    st.header("🔮 Predict Project Delay")
    st.caption("Fill in the project details below, exactly as they would appear at planning time.")

    numeric_features = artifacts["numeric_features"]
    categorical_options = artifacts["categorical_options"]
    numeric_stats = artifacts["numeric_stats"]

    with st.form("prediction_form"):
        st.markdown("#### Project scale & ownership")
        c1, c2, c3 = st.columns(3)
        raw_inputs = {}
        numeric_layout = {
            c1: ["Land_Area_Acres", "Number_of_Owners", "Number_of_Parcels", "Affected_Families"],
            c2: ["Legal_Cases", "Ownership_Disputes", "Documents_Missing", "Objections_Received"],
            c3: ["Government_Clearances", "Planned_Duration_Days", "Approval_Days"],
        }
        labels = {
            "Land_Area_Acres": "Land Area (Acres)",
            "Number_of_Owners": "Number of Owners",
            "Number_of_Parcels": "Number of Parcels",
            "Affected_Families": "Affected Families",
            "Legal_Cases": "Pending Legal Cases",
            "Ownership_Disputes": "Ownership Disputes",
            "Documents_Missing": "Documents Missing",
            "Objections_Received": "Objections Received",
            "Government_Clearances": "Government Clearances Obtained",
            "Planned_Duration_Days": "Planned Duration (Days)",
            "Approval_Days": "Approval Time (Days)",
        }

        for col, features in numeric_layout.items():
            with col:
                for feature in features:
                    stats = numeric_stats[feature]
                    is_integerish = feature not in ("Land_Area_Acres",)
                    if is_integerish:
                        raw_inputs[feature] = st.number_input(
                            labels[feature],
                            min_value=0,
                            value=int(round(stats["median"])),
                            step=1,
                            key=f"num_{feature}",
                        )
                    else:
                        raw_inputs[feature] = st.number_input(
                            labels[feature],
                            min_value=0.0,
                            value=float(round(stats["median"], 2)),
                            step=0.5,
                            key=f"num_{feature}",
                        )

        st.markdown("#### Project classification")
        c4, c5, c6 = st.columns(3)
        with c4:
            raw_inputs["Project_Type"] = st.selectbox(
                "Project Type", categorical_options["Project_Type"]["options"]
            )
        with c5:
            raw_inputs["Location_Zone"] = st.selectbox(
                "Location Zone", categorical_options["Location_Zone"]["options"]
            )
        with c6:
            raw_inputs["Land_Type"] = st.selectbox(
                "Land Type", categorical_options["Land_Type"]["options"]
            )

        st.markdown("#### Risk factors")
        c7, c8, c9 = st.columns(3)
        with c7:
            raw_inputs["Rehabilitation_Required"] = st.selectbox(
                "Rehabilitation Required", categorical_options["Rehabilitation_Required"]["options"]
            )
        with c8:
            raw_inputs["Compensation_Dispute"] = st.selectbox(
                "Compensation Dispute", categorical_options["Compensation_Dispute"]["options"]
            )
        with c9:
            raw_inputs["Previous_Acquisition_Delay"] = st.selectbox(
                "Previous Acquisition Delay", categorical_options["Previous_Acquisition_Delay"]["options"]
            )

        submitted = st.form_submit_button("🔍 Run Prediction", use_container_width=True)

    if not submitted:
        return

    errors = []
    for feature in numeric_features:
        val = raw_inputs.get(feature)
        if val is None:
            errors.append(f"Missing value for {labels.get(feature, feature)}.")
        elif val < 0:
            errors.append(f"{labels.get(feature, feature)} cannot be negative.")

    if errors:
        for e in errors:
            st.error(e)
        return

    try:
        prediction, probability = predict_delay(raw_inputs, artifacts)
    except Exception as e:
        st.error(
            "Something went wrong while running the prediction. "
            "Please check your inputs and try again."
        )
        with st.expander("Technical details"):
            st.code(str(e))
        return

    prob_delay = float(probability[1])
    prob_no_delay = float(probability[0])

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-card result-delayed">
                <div class="headline">⚠️ Prediction: Project Likely to be DELAYED</div>
                <div class="subline">Confidence: {prob_delay*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-ontime">
                <div class="headline">✅ Prediction: Project Likely ON TIME</div>
                <div class="subline">Confidence: {prob_no_delay*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Prediction probabilities")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.metric("Probability: Not Delayed", f"{prob_no_delay*100:.2f}%")
        st.progress(prob_no_delay)
    with pc2:
        st.metric("Probability: Delayed", f"{prob_delay*100:.2f}%")
        st.progress(prob_delay)


def render_model_performance(artifacts: dict):
    st.header("📊 Model Performance")
    st.caption("Metrics below are computed directly from the original notebook's evaluation cells.")

    metrics = artifacts["metrics"]

    st.subheader("Deployed model")
    deployed = metrics["deployed_final_model"]
    d1, d2, d3 = st.columns(3)
    d1.markdown(f'<div class="metric-card"><div class="label">Algorithm</div><div class="value" style="font-size:1.1rem;">{deployed["algorithm"]}</div></div>', unsafe_allow_html=True)
    d2.markdown(f'<div class="metric-card"><div class="label">Test Accuracy</div><div class="value">{deployed["accuracy"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    d3.markdown(f'<div class="metric-card"><div class="label">Test Set Size</div><div class="value">{sum(sum(row) for row in deployed["confusion_matrix"])}</div></div>', unsafe_allow_html=True)
    st.caption(deployed["note"])

    cm = deployed["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual: Not Delayed", "Actual: Delayed"], columns=["Pred: Not Delayed", "Pred: Delayed"])
    st.markdown("**Confusion Matrix**")
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("**Classification Report**")
    st.dataframe(pd.DataFrame(deployed["classification_report"]).transpose(), use_container_width=True)

    st.divider()
    st.subheader("Model comparison (from notebook development)")
    chart_data = metrics["notebook_comparison_chart"]
    chart_df = pd.DataFrame({"Accuracy": chart_data["accuracy"]}, index=chart_data["models"])
    st.bar_chart(chart_df)

    with st.expander("Logistic Regression — detailed development metrics (cells 21-25)"):
        lr = metrics["logistic_regression"]
        st.write(f"Accuracy: **{lr['accuracy']*100:.1f}%**")
        lr_cm = pd.DataFrame(lr["confusion_matrix"], index=["Actual: Not Delayed", "Actual: Delayed"], columns=["Pred: Not Delayed", "Pred: Delayed"])
        st.dataframe(lr_cm, use_container_width=True)
        st.dataframe(pd.DataFrame(lr["classification_report"]).transpose(), use_container_width=True)

    with st.expander("Decision Tree — development accuracy (cell 26)"):
        st.write(f"Accuracy: **{metrics['decision_tree']['accuracy']*100:.1f}%**")

    with st.expander("Random Forest — detailed development metrics + feature importance (cells 27-29)"):
        rf = metrics["random_forest"]
        st.write(f"Accuracy: **{rf['accuracy']*100:.1f}%**")
        st.dataframe(pd.DataFrame(rf["classification_report"]).transpose(), use_container_width=True)
        st.markdown("**Feature Importance**")
        fi_df = pd.DataFrame(rf["feature_importance"]).set_index("Feature")
        st.bar_chart(fi_df)


def render_about():
    st.header("ℹ️ About This Project")
    st.write(
        "**Land Acquisition Delay Predictor** is a machine learning project that "
        "estimates the likelihood of delay in land acquisition projects, using "
        "historical project data (ownership complexity, legal disputes, missing "
        "documents, government clearances, and planned timelines)."
    )
    st.markdown("#### Architecture")
    st.code(
        "USER\n"
        "  -> STREAMLIT FRONTEND (this app)\n"
        "  -> INPUT VALIDATION\n"
        "  -> EXISTING PREPROCESSING (StandardScaler + one-hot encoding, from the notebook)\n"
        "  -> EXISTING TRAINED MODEL (LogisticRegression, from the notebook)\n"
        "  -> PREDICTION\n"
        "  -> STREAMLIT RESULT",
        language="text",
    )
    st.markdown("#### Reused from the original notebook")
    st.markdown(
        "- Data cleaning steps (median/mode imputation, duplicate removal)\n"
        "- `StandardScaler` fitted on the same numeric columns\n"
        "- `pd.get_dummies(drop_first=True)` one-hot encoding of the same categorical columns\n"
        "- The trained `LogisticRegression(max_iter=1000)` final model\n"
        "- The `predict_with_user_input` prediction logic (adapted from CLI input to a web form)"
    )
    st.caption("This app does not retrain, replace, or modify the original model or preprocessing.")


def main():
    st.sidebar.title("🏗️ Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "🔮 Prediction", "📊 Model Performance", "ℹ️ About Project"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption("Land Acquisition Delay Predictor")
    st.sidebar.caption("Backend model: Logistic Regression (scikit-learn)")

    try:
        artifacts = load_artifacts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Run this in your terminal, from the project folder:\n\n"
            "```\npython train_and_export.py --csv data/land_acquisition_delay_synthetic.csv\n```"
        )
        st.stop()
    except Exception as e:
        st.error("Failed to load the model artifacts.")
        with st.expander("Technical details"):
            st.code(str(e))
        st.stop()

    if page == "🏠 Home":
        render_home()
    elif page == "🔮 Prediction":
        render_prediction(artifacts)
    elif page == "📊 Model Performance":
        render_model_performance(artifacts)
    else:
        render_about()


if __name__ == "__main__":
    main()
