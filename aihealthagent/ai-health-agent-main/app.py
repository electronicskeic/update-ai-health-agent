from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from health_agent.chatbot import ChatContext, answer
from health_agent.data import bmi_category, calculate_bmi, load_dataset, summarize_dataset, to_model_frame, load_diabetes_dataset, summarize_diabetes_dataset, to_diabetes_model_frame
from health_agent.model import evaluate_model, predict_obesity_risk, train_obesity_model, train_diabetes_model, predict_diabetes_risk, evaluate_diabetes_model
from health_agent.llm import chat_with_llm, get_llm_config
from health_agent.recommendations import build_plan
from health_agent.storage import (
    add_chat_message,
    add_checkin,
    authenticate,
    create_user,
    init_db,
    list_chat_messages,
    list_checkins,
    load_profile,
    upsert_profile,
 )


APP_DIR = Path(__file__).parent
DATASET_PATH = APP_DIR / "bmi.xlsx"
DATASET_DIABETES_PATH = APP_DIR / "diabetes.csv"
DB_PATH = APP_DIR / "health_agent.db"


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_dataset(DATASET_PATH)


@st.cache_data(show_spinner=False)
def get_summary() -> dict:
    df = get_data()
    s = summarize_dataset(df)
    return {
        "n": s.n,
        "bmi_mean": s.bmi_mean,
        "bmi_median": s.bmi_median,
        "overweight_rate": s.overweight_rate,
        "obese_rate": s.obese_rate,
        "age_bmi_corr": s.age_bmi_corr,
        "bmi_bins": s.bmi_bins,
    }


@st.cache_resource(show_spinner=False)
def get_model():
    df = get_data()
    df_model = to_model_frame(df)
    return train_obesity_model(df_model)


@st.cache_data(show_spinner=False)
def get_diabetes_data() -> pd.DataFrame:
    # Force cache reload by changing source slightly
    return load_diabetes_dataset(DATASET_DIABETES_PATH)


@st.cache_resource(show_spinner=False)
def get_diabetes_model():
    df = get_diabetes_data()
    df_model = to_diabetes_model_frame(df)
    return train_diabetes_model(df_model)


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def lb_to_kg(lb: float) -> float:
    return float(lb) * 0.45359237


def inch_to_m(inch: float) -> float:
    return float(inch) * 0.0254


def kg_to_lb(kg: float) -> float:
    return float(kg) / 0.45359237


st.set_page_config(page_title="AI Health Agent", layout="wide")
st.title("AI Health Agent")
st.caption("Startup-style health agent for BMI, risk insights, and actionable plans (dataset-driven).")

init_db(DB_PATH)
llm_cfg = get_llm_config()

with st.sidebar:
    st.markdown("### Navigation")
    if "user" not in st.session_state:
        st.session_state["user"] = None

    user = st.session_state.get("user")
    if user is None:
        auth_mode = st.radio("Account", ["Log in", "Sign up"], horizontal=True)
        username = st.text_input("Username", placeholder="e.g., alex", key="auth_username")
        password = st.text_input("Password", type="password", key="auth_password")
        if auth_mode == "Sign up":
            password2 = st.text_input("Confirm password", type="password", key="auth_password2")
        if st.button(auth_mode, use_container_width=True):
            try:
                if auth_mode == "Sign up":
                    if password != password2:
                        st.error("Passwords do not match.")
                    else:
                        u = create_user(DB_PATH, username=username, password=password)
                        st.session_state["user"] = u
                        st.success("Account created.")
                        st.rerun()
                else:
                    u = authenticate(DB_PATH, username=username, password=password)
                    if not u:
                        st.error("Invalid username/password.")
                    else:
                        st.session_state["user"] = u
                        st.success("Logged in.")
                        st.rerun()
            except Exception as e:
                st.error(str(e))

        st.divider()
        st.markdown("### Safety")
        st.info("Educational only. Not medical advice. If you have symptoms or concerns, consult a clinician.")
        st.stop()

    st.success(f"Logged in as **{user.username}**")
    if st.button("Log out", use_container_width=True):
        st.session_state["user"] = None
        st.rerun()

    page = st.radio("Go to", ["Onboarding", "My Plan", "Diabetes Risk", "Insights", "Report", "Chat"], index=0, label_visibility="collapsed")
    st.divider()
    st.markdown("### Chat mode")
    if llm_cfg:
        st.write(f"LLM enabled: **{llm_cfg.model}**")
        use_llm = st.toggle("Use LLM for chat", value=True)
    else:
        st.write("LLM disabled (set `GEMINI_API_KEY` to enable).")
        use_llm = False

    st.divider()
    st.markdown("### Safety")
    st.info("Educational only. Not medical advice. If you have symptoms or concerns, consult a clinician.")

if not DATASET_PATH.exists():
    st.error(f"Missing dataset: {DATASET_PATH}")
    st.stop()

df = get_data()
summary = summarize_dataset(df)
model = get_model()
df_model = to_model_frame(df)

df_diabetes = get_diabetes_data()
summary_diabetes = summarize_diabetes_dataset(df_diabetes)
model_diabetes = get_diabetes_model()
df_diabetes_model = to_diabetes_model_frame(df_diabetes)

colA, colB, colC, colD = st.columns(4)
colA.metric("Dataset size", f"{summary.n}")
colB.metric("BMI mean", f"{summary.bmi_mean:.2f}")
colC.metric("Overweight prevalence (BMI≥25)", pct(summary.overweight_rate))
colD.metric("Obesity prevalence (BMI≥30)", pct(summary.obese_rate))

if "profile" not in st.session_state:
    st.session_state["profile"] = {}

if "checkins" not in st.session_state:
    st.session_state["checkins"] = []

u = st.session_state["user"]
loaded = load_profile(DB_PATH, user_id=u.id)
if loaded is None:
    loaded = {
        "name": "",
        "goal": "Lose weight",
        "units": "Metric",
        "diet_pref": "No preference",
        "activity": "Moderate (3–5 days/week)",
        "constraints": "",
    }
st.session_state["profile"] = loaded
st.session_state["checkins"] = list_checkins(DB_PATH, user_id=u.id)


def set_latest(*, age: int, height_m: float, weight_kg: float):
    bmi = calculate_bmi(height_m, weight_kg)
    cat = bmi_category(bmi)
    risk = predict_obesity_risk(model, age=age, height_m=height_m, weight_kg=weight_kg, bmi=bmi)
    plan = build_plan(bmi=bmi, category=cat, risk_level=risk.risk_level)
    st.session_state["last_bmi"] = bmi
    st.session_state["last_category"] = cat
    st.session_state["last_risk"] = risk
    st.session_state["last_plan"] = plan


if page == "Onboarding":
    st.subheader("Onboarding")
    st.write("Set your profile once, then use **My Plan** to get an actionable weekly plan.")

    p = st.session_state["profile"]
    c1, c2 = st.columns(2)
    p["name"] = c1.text_input("Name (optional)", value=p.get("name", ""))
    p["goal"] = c2.selectbox("Goal", ["Lose weight", "Maintain weight", "Gain weight"], index=["Lose weight", "Maintain weight", "Gain weight"].index(p.get("goal", "Lose weight")))

    c3, c4 = st.columns(2)
    p["units"] = c3.selectbox("Units", ["Metric", "Imperial"], index=0 if p.get("units") == "Metric" else 1)
    p["activity"] = c4.selectbox(
        "Activity level",
        ["Low (0–2 days/week)", "Moderate (3–5 days/week)", "High (6–7 days/week)"],
        index=["Low (0–2 days/week)", "Moderate (3–5 days/week)", "High (6–7 days/week)"].index(p.get("activity", "Moderate (3–5 days/week)")),
    )

    c5, c6 = st.columns(2)
    p["diet_pref"] = c5.selectbox("Diet preference", ["No preference", "Vegetarian", "Vegan", "Halal", "Keto-ish", "High-protein"], index=["No preference", "Vegetarian", "Vegan", "Halal", "Keto-ish", "High-protein"].index(p.get("diet_pref", "No preference")))
    p["constraints"] = c6.text_input("Constraints (allergies, injuries, schedule)", value=p.get("constraints", ""))

    st.divider()
    st.markdown("**Your current measurements**")
    a1, a2, a3 = st.columns(3)
    age = int(a1.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1))

    if p["units"] == "Metric":
        height_m = float(a2.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=1.70, step=0.01, format="%.2f"))
        weight_kg = float(a3.number_input("Weight (kg)", min_value=10.0, max_value=400.0, value=70.0, step=0.1, format="%.1f"))
    else:
        height_in = float(a2.number_input("Height (inches)", min_value=20.0, max_value=100.0, value=67.0, step=0.5, format="%.1f"))
        weight_lb = float(a3.number_input("Weight (lb)", min_value=22.0, max_value=900.0, value=154.0, step=0.5, format="%.1f"))
        height_m = inch_to_m(height_in)
        weight_kg = lb_to_kg(weight_lb)

    if st.button("Save & Generate my plan", type="primary"):
        set_latest(age=age, height_m=height_m, weight_kg=weight_kg)
        upsert_profile(DB_PATH, user_id=u.id, profile=p)
        st.success("Saved. Go to **My Plan** for your personalized plan and next steps.")


elif page == "My Plan":
    st.subheader("My Plan")
    p = st.session_state["profile"]
    last_bmi = st.session_state.get("last_bmi")
    last_risk = st.session_state.get("last_risk")
    last_plan = st.session_state.get("last_plan")
    last_cat = st.session_state.get("last_category")

    if last_bmi is None or last_plan is None or last_risk is None:
        st.warning("Complete **Onboarding** first so I can tailor your plan.")
    else:
        top1, top2, top3 = st.columns(3)
        top1.metric("BMI", f"{last_bmi:.1f}")
        top2.metric("Category", str(last_cat))
        top3.metric("Obesity risk (dataset-based)", f"{last_risk.obesity_probability:.2f} ({last_risk.risk_level})")

        st.markdown("### This week’s focus")
        st.write(f"**Goal:** {p.get('goal','')}  |  **Activity:** {p.get('activity','')}  |  **Diet:** {p.get('diet_pref','')}")
        if p.get("constraints"):
            st.write(f"**Constraints:** {p['constraints']}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Diet (pick 1–2 to start)**")
            for x in last_plan.diet:
                st.write(f"- {x}")
        with right:
            st.markdown("**Fitness (pick 1–2 to start)**")
            for x in last_plan.fitness:
                st.write(f"- {x}")

        if last_plan.notes:
            st.markdown("**Notes**")
            for n in last_plan.notes:
                st.write(f"- {n}")

        st.divider()
        st.markdown("### Progress tracking (in-session)")
        c1, c2, c3 = st.columns(3)
        date = c1.date_input("Check-in date")
        if p.get("units") == "Imperial":
            w = float(c2.number_input("Weight (lb)", min_value=22.0, max_value=900.0, value=kg_to_lb(70.0), step=0.5, format="%.1f"))
            wkg = lb_to_kg(w)
        else:
            wkg = float(c2.number_input("Weight (kg)", min_value=10.0, max_value=400.0, value=70.0, step=0.1, format="%.1f"))
        note = c3.text_input("Note (optional)", value="")

        if st.button("Add check-in"):
            add_checkin(DB_PATH, user_id=u.id, date=str(date), weight_kg=float(wkg), note=note or None)
            st.session_state["checkins"] = list_checkins(DB_PATH, user_id=u.id)
            st.success("Check-in added.")

        if st.session_state["checkins"]:
            ch = pd.DataFrame(st.session_state["checkins"])
            ch["date"] = pd.to_datetime(ch["date"])
            ch = ch.sort_values("date")
            fig = px.line(ch, x="date", y="weight_kg", markers=True, title="Weight over time (kg)")
            st.plotly_chart(fig, use_container_width=True)


elif page == "Diabetes Risk":
    st.subheader("Diabetes Risk Prediction")
    st.write("Enter your clinical metrics to estimate diabetes risk based on the Pima Indians diabetes dataset.")
    
    col1, col2 = st.columns(2)
    preg = col1.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
    glucose = col2.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
    bp = col1.number_input("Blood Pressure (Diastolic)", min_value=0.0, max_value=200.0, value=70.0)
    skin = col2.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
    insulin = col1.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=79.0)
    bmi_input = col2.number_input("BMI", min_value=0.0, max_value=70.0, value=st.session_state.get("last_bmi", 25.0))
    dpf = col1.number_input("Diabetes Pedigree Function", min_value=0.001, max_value=3.0, value=0.5)
    age_input = col2.number_input("Age", min_value=1, max_value=120, value=30)
    
    if st.button("Predict Diabetes Risk", type="primary"):
        risk = predict_diabetes_risk(model_diabetes, pregnancies=preg, glucose=glucose, bp=bp, skin=skin, insulin=insulin, bmi=bmi_input, dpf=dpf, age=age_input)
        st.session_state["last_diabetes_risk"] = risk
        
    dr = st.session_state.get("last_diabetes_risk")
    if dr:
        st.success(f"**Risk Level:** {dr.risk_level}  \n**Probability:** {dr.diabetes_probability:.1%}")


elif page == "Insights":
    st.subheader("Insights")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**BMI distribution**")
        fig = px.histogram(df, x="BMI", nbins=20, title="BMI Histogram")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Age vs BMI**")
        fig2 = px.scatter(df, x="Age", y="BMI", trendline="ols", title="Age vs BMI (trendline)")
        st.plotly_chart(fig2, use_container_width=True)
        r = summary.age_bmi_corr
        strength = "weak" if abs(r) < 0.3 else ("moderate" if abs(r) < 0.6 else "strong")
        st.write(f"Correlation (Age↔BMI): **{r:.3f}** → typically interpreted as **{strength}**.")

    st.markdown("**BMI class breakdown**")
    class_counts = df["BMI Class"].astype(str).value_counts().reset_index()
    class_counts.columns = ["BMI Class", "Count"]
    fig3 = px.bar(class_counts, x="BMI Class", y="Count", title="BMI Classes (dataset labels)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Model transparency")
    q = evaluate_model(df_model, folds=5)
    st.write(f"- Uses features: **Age, Height, Weight, BMI**")
    st.write(f"- Cross-validated ROC-AUC on this dataset: **{q.roc_auc_mean:.3f} ± {q.roc_auc_std:.3f}**")
    st.caption("Risk is dataset-based and may not generalize to all populations.")

    st.divider()
    st.markdown("### Diabetes Data Insights")
    q2 = evaluate_diabetes_model(df_diabetes_model, folds=5)
    st.write(f"- Cross-validated ROC-AUC on Pima Diabetes dataset: **{q2.roc_auc_mean:.3f} ± {q2.roc_auc_std:.3f}**")
    
    c3, c4 = st.columns(2)
    with c3:
        fig_d1 = px.histogram(df_diabetes, x="Glucose", color="Outcome", title="Glucose by Diabetes Outcome")
        st.plotly_chart(fig_d1, use_container_width=True)
    with c4:
        fig_d2 = px.histogram(df_diabetes, x="BMI", color="Outcome", title="BMI by Diabetes Outcome")
        st.plotly_chart(fig_d2, use_container_width=True)


elif page == "Report":
    st.subheader("Report")
    p = st.session_state["profile"]
    last_bmi = st.session_state.get("last_bmi")
    last_risk = st.session_state.get("last_risk")
    last_plan = st.session_state.get("last_plan")
    last_cat = st.session_state.get("last_category")

    if last_bmi is None or last_risk is None or last_plan is None:
        st.warning("Complete **Onboarding** first to generate your report.")
    else:
        report = (
            f"# Personal Health Summary\n\n"
            f"**Name:** {p.get('name') or 'N/A'}\n\n"
            f"**Goal:** {p.get('goal')}\n\n"
            f"## Key numbers\n\n"
            f"- BMI: **{last_bmi:.1f}**\n"
            f"- Category: **{last_cat}**\n"
            f"- Obesity risk (dataset-based): **{last_risk.obesity_probability:.2f} ({last_risk.risk_level})**\n\n"
            f"## Recommendations\n\n"
            f"### Diet\n" + "\n".join([f"- {x}" for x in last_plan.diet]) + "\n\n"
            f"### Fitness\n" + "\n".join([f"- {x}" for x in last_plan.fitness]) + "\n\n"
            f"### Notes\n" + "\n".join([f"- {x}" for x in (last_plan.notes or ['None'])]) + "\n\n"
            f"## Population comparison (dataset)\n\n"
            f"- Overweight prevalence (BMI≥25): **{pct(summary.overweight_rate)}**\n"
            f"- Obesity prevalence (BMI≥30): **{pct(summary.obese_rate)}**\n\n"
            f"## Disclaimer\n\n"
            f"Educational only; not medical advice.\n"
        )
        st.markdown(report)
        st.download_button("Download report (Markdown)", data=report.encode("utf-8"), file_name="health_report.md", mime="text/markdown")


elif page == "Chat":
    st.subheader("Chat")
    st.caption("Ask questions naturally. The agent uses your dataset insights + your latest saved profile/measurements.")

    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    if not st.session_state["chat"]:
        st.session_state["chat"] = [
            ("Agent", "Hi! Ask me about your next step, diet, fitness, risk, or what your numbers mean.")
        ]

    if st.button("Load my saved chat", use_container_width=True):
        saved = list_chat_messages(DB_PATH, user_id=u.id, limit=30)
        st.session_state["chat"] = []
        for m in saved:
            st.session_state["chat"].append(("You" if m["role"] == "user" else "Agent", m["content"]))

    user_msg = st.text_input("Message", placeholder="e.g., What should I do next?")
    if st.button("Send", type="primary", use_container_width=True) and user_msg.strip():
        user_text = user_msg.strip()
        st.session_state["chat"].append(("You", user_text))
        add_chat_message(DB_PATH, user_id=u.id, role="user", content=user_text)

        ctx = ChatContext(
            summary=summary,
            profile=st.session_state.get("profile"),
            last_bmi=st.session_state.get("last_bmi"),
            last_risk=st.session_state.get("last_risk"),
            last_plan=st.session_state.get("last_plan"),
        )

        if use_llm and llm_cfg:
            system_prompt = (
                "You are a practical health coach inside a BMI app. "
                "Be concise, action-oriented, and grounded in the provided dataset stats and the user's latest saved profile. "
                "Do not claim to diagnose disease. If asked for medical advice, recommend a clinician.\n\n"
                f"Dataset: n={summary.n}, BMI_mean={summary.bmi_mean:.2f}, overweight_rate={pct(summary.overweight_rate)}, "
                f"obesity_rate={pct(summary.obese_rate)}, age_bmi_corr={summary.age_bmi_corr:.3f}.\n"
                f"Diabetes Dataset: n={summary_diabetes.n}, positive_rate={pct(summary_diabetes.positive_rate)}.\n"
                f"User profile: {st.session_state.get('profile')}\n"
                f"Latest BMI: {st.session_state.get('last_bmi')}\n"
                f"Latest obesity risk: {st.session_state.get('last_risk')}\n"
                f"Latest diabetes risk: {st.session_state.get('last_diabetes_risk')}\n"
            )
            history = []
            for speaker, msg in st.session_state["chat"][-12:]:
                history.append({"role": "user" if speaker == "You" else "assistant", "content": msg})
            try:
                reply = chat_with_llm(system_prompt=system_prompt, messages=history, model=llm_cfg.model).strip()
                if not reply:
                    reply = answer(user_text, ctx)
            except Exception:
                reply = answer(user_text, ctx)
        else:
            reply = answer(user_text, ctx)

        st.session_state["chat"].append(("Agent", reply))
        add_chat_message(DB_PATH, user_id=u.id, role="assistant", content=reply)

    for speaker, msg in st.session_state["chat"]:
        st.markdown(f"**{speaker}:** {msg}")

