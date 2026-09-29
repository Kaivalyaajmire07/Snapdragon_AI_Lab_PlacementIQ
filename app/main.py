import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import shap

st.set_page_config(page_title="PlacementIQ", page_icon="🎓", layout="wide")

FEATURES = [
    "attendance", "cgpa", "technical_score", "projects",
    "internships", "communication", "backlogs"
]

TARGET = "placement_readiness"

@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "students.csv")
    return pd.read_csv(path)

@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]
    model = RandomForestRegressor(
        n_estimators=250, random_state=42, min_samples_leaf=1
    )
    model.fit(X, y)
    return model

@st.cache_resource
def cluster_model(df):
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(df[["cgpa", "technical_score", "projects", "communication"]])
    return km, labels

def score_band(score):
    if score >= 80:
        return "High Readiness", "Strong placement profile"
    if score >= 65:
        return "Moderate Readiness", "Good base with improvement areas"
    return "Needs Improvement", "Focus on the recommended skill gaps"

def recommendations(row):
    recs = []
    if row.technical_score < 70:
        recs.append("Strengthen Python, SQL and practical data-analysis skills.")
    if row.projects < 3:
        recs.append("Build 1–2 portfolio projects with measurable outcomes.")
    if row.internships < 1:
        recs.append("Add a relevant internship, virtual internship or industry project.")
    if row.communication < 7:
        recs.append("Practice technical communication, interviews and presentations.")
    if row.cgpa < 8:
        recs.append("Maintain/improve academic performance while building practical skills.")
    if row.attendance < 80:
        recs.append("Improve attendance and consistency in academic work.")
    if row.backlogs > 0:
        recs.append("Prioritize clearing backlogs because they can affect eligibility.")
    return recs or ["Maintain your current profile and continue targeted interview preparation."]

def assistant_answer(question, row, pred, recs):
    q = question.lower()
    if any(x in q for x in ["why", "score", "readiness", "low"]):
        return (
            f"Your predicted placement readiness is **{pred:.0f}%**. "
            f"The strongest improvement areas are based on your technical score "
            f"({row.technical_score}/100), projects ({row.projects}), "
            f"internships ({row.internships}), communication ({row.communication}/10), "
            f"CGPA ({row.cgpa}) and backlogs ({row.backlogs}). "
            f"Start with: {recs[0]}"
        )
    if any(x in q for x in ["skill", "learn", "improve", "next"]):
        return "Recommended next steps:\n\n" + "\n".join([f"- {x}" for x in recs])
    if "project" in q:
        return (
            "A useful next project should match your target role and demonstrate "
            "real data collection, cleaning, analysis and deployment. Add a measurable "
            "business outcome and publish the work with a clear README."
        )
    if "interview" in q:
        return (
            "Prepare a focused interview plan: Python/SQL fundamentals, ML concepts, "
            "one end-to-end project explanation, STAR-format behavioral answers and "
            "a short explanation of your placement-readiness model."
        )
    return (
        f"Based on the current profile, the readiness prediction is **{pred:.0f}%**. "
        "Ask me about your score, skills, projects or interview preparation."
    )

df = load_data()
model = train_model(df)
km, labels = cluster_model(df)
df["cluster"] = labels

st.sidebar.title("🎓 PlacementIQ")
st.sidebar.caption("Snapdragon AI Lab Demo")
page = st.sidebar.radio("Navigate", ["Dashboard", "AI Placement Assistant", "Explainability", "Analytics", "About"])

if page == "Dashboard":
    st.title("PlacementIQ")
    st.subheader("AI-Powered Student Placement Performance Analytics")
    st.info("Local-first prototype: ML predictions and explanations run on the computer. No external API key is required.")

    selected = st.selectbox("Select student", df["name"].tolist())
    row = df[df.name == selected].iloc[0]
    pred = float(model.predict(pd.DataFrame([row[FEATURES].values], columns=FEATURES))[0])
    band, desc = score_band(pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Placement Readiness", f"{pred:.0f}%")
    c2.metric("CGPA", f"{row.cgpa:.1f}")
    c3.metric("Technical Score", f"{row.technical_score}/100")
    c4.metric("Projects", int(row.projects))

    st.success(f"**{band}** — {desc}")

    left, right = st.columns(2)
    with left:
        st.markdown("### Student Profile")
        profile = pd.DataFrame({
            "Metric": ["Attendance", "CGPA", "Technical Score", "Projects", "Internships", "Communication", "Backlogs"],
            "Value": [
                f"{row.attendance}%", row.cgpa, f"{row.technical_score}/100",
                row.projects, row.internships, f"{row.communication}/10", row.backlogs
            ]
        })
        st.dataframe(profile, hide_index=True, use_container_width=True)

    with right:
        st.markdown("### Skill-Gap Recommendations")
        for item in recommendations(row):
            st.write("• " + item)

elif page == "AI Placement Assistant":
    st.title("🤖 AI Placement Assistant")
    st.caption("A local recommendation/explanation layer over the student's ML results. Replace with a local LLM for the final Snapdragon AI-PC build.")

    selected = st.selectbox("Student", df["name"].tolist())
    row = df[df.name == selected].iloc[0]
    pred = float(model.predict(pd.DataFrame([row[FEATURES].values], columns=FEATURES))[0])
    recs = recommendations(row)

    st.metric("Current readiness prediction", f"{pred:.0f}%")
    question = st.text_input(
        "Ask a question",
        placeholder="Why is my readiness score low? What should I learn next?"
    )

    if question:
        st.markdown("### Assistant")
        st.markdown(assistant_answer(question, row, pred, recs))

    st.markdown("### Try these questions")
    for q in [
        "Why is my readiness score low?",
        "What should I learn next?",
        "How can I improve my projects?",
        "How should I prepare for interviews?"
    ]:
        st.write("• " + q)

elif page == "Explainability":
    st.title("🔎 Explainable AI — SHAP")
    selected = st.selectbox("Student", df["name"].tolist())
    row = df[df.name == selected].iloc[0]
    X = df[FEATURES]
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)

    idx = df.index[df.name == selected][0]
    values = sv.values[idx]
    exp_df = pd.DataFrame({"Feature": FEATURES, "Impact": values})
    exp_df["Direction"] = np.where(exp_df.Impact >= 0, "Increases prediction", "Decreases prediction")
    exp_df = exp_df.sort_values("Impact")

    fig = px.bar(
        exp_df, x="Impact", y="Feature", color="Direction",
        orientation="h", title="Feature contribution for selected student"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("SHAP explains how model features influence the Random Forest prediction. It is an explanation of the model, not a causal claim.")

elif page == "Analytics":
    st.title("📊 Cohort Analytics")
    a, b = st.columns(2)
    with a:
        fig = px.scatter(
            df, x="cgpa", y="technical_score", size="projects",
            color="placement_readiness", hover_name="name",
            title="Academic vs Technical Profile"
        )
        st.plotly_chart(fig, use_container_width=True)
    with b:
        counts = df["cluster"].value_counts().sort_index().rename("students").reset_index()
        counts.columns = ["cluster", "students"]
        fig = px.bar(counts, x="cluster", y="students", title="K-Means Student Groups")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df[["name", "cgpa", "technical_score", "projects", "internships", "placement_readiness", "cluster"]],
        hide_index=True, use_container_width=True
    )

elif page == "About":
    st.title("About the Challenge Build")
    st.markdown("""
### Snapdragon AI Lab — Build & Present

**Goal:** demonstrate an AI use case designed for a Snapdragon-powered HP PC.

**Current prototype components**
- Random Forest placement-readiness prediction
- K-Means student grouping
- SHAP explainability
- Personalized skill-gap recommendations
- Local-first AI assistant interface

**Final Snapdragon step**
Use a supported open-source model / Qualcomm AI Hub deployment artifact for local generative inference on the target Snapdragon HP PC, then capture evidence of the actual device deployment.

**Privacy angle**
The application is designed around local processing of student information, reducing dependence on external cloud inference.
""")
