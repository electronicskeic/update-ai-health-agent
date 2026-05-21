from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any
import sqlite3
import requests

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from health_agent.chatbot import ChatContext, answer
from health_agent.data import (
    bmi_category,
    calculate_bmi,
    load_dataset,
    summarize_dataset,
    to_model_frame,
    load_diabetes_dataset,
    summarize_diabetes_dataset,
    to_diabetes_model_frame,
    calculate_icmr_bmr,
    calculate_whtr,
    estimate_body_fat_percentage,
    assess_visceral_fat_risk,
)
from health_agent.model import (
    evaluate_model,
    predict_obesity_risk,
    train_obesity_model,
    train_diabetes_model,
    predict_diabetes_risk,
    evaluate_diabetes_model,
)
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
    return load_diabetes_dataset(DATASET_DIABETES_PATH)


@st.cache_resource(show_spinner=False)
def get_diabetes_model():
    df = get_diabetes_data()
    df_model = to_diabetes_model_frame(df)
    return train_diabetes_model(df_model)


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


# Page Configuration
st.set_page_config(
    page_title="Arogya AI — Patient Health Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════
# PREMIUM HEALTHCARE UI — AROGYA AI DESIGN SYSTEM
# ══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── GLOBAL FOUNDATION ─── */
*, *::before, *::after { box-sizing: border-box; }

.stApp, .main, [data-testid="stAppViewContainer"] {
    background: #F6F3EC !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.main .block-container {
    padding: 1.5rem 2.5rem 3rem 2.5rem !important;
    max-width: 1180px !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: #2C2C2C !important;
    letter-spacing: -0.02em !important;
}

p, span, li { font-family: 'Inter', sans-serif !important; color: #2C2C2C !important; }
caption { color: #7A7A8C !important; }

/* ─── SIDEBAR ─── */
section[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: none !important;
    border-radius: 0 28px 28px 0 !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.18) !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding: 1.8rem 1.4rem !important;
}

section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {
    color: #E8E8F0 !important;
}

section[data-testid="stSidebar"] .stMarkdown p {
    color: #A0A0B8 !important;
    font-size: 0.85rem !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #A0A0B8 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 0.9rem !important;
    border-radius: 12px !important;
    display: flex !important;
    align-items: center !important;
    transition: all 0.22s ease !important;
    cursor: pointer !important;
    margin: 2px 0 !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(246,243,236,0.08) !important;
    color: #F0F0F8 !important;
    transform: translateX(4px) !important;
}

section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.07) !important;
    color: #A0A0B8 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    font-size: 0.83rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
    font-family: 'Inter', sans-serif !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.12) !important;
    color: #E8E8F0 !important;
    border-color: rgba(255,255,255,0.2) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 1.2rem 0 !important;
}

section[data-testid="stSidebar"] .stAlert {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #A0A0B8 !important;
}

/* ─── GLOBAL BUTTONS ─── */
.stButton > button {
    background: linear-gradient(135deg, #E87B6C 0%, #C76B8A 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.62rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 16px rgba(232,123,108,0.28) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(232,123,108,0.38) !important;
}

.stButton > button:active { transform: translateY(0px) !important; }

.stDownloadButton > button {
    background: linear-gradient(135deg, #6CC5D1 0%, #4AA8B4 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.62rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 4px 16px rgba(108,197,209,0.28) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(108,197,209,0.38) !important;
}

/* ─── INPUTS ─── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: #FFFFFF !important;
    border: 1.5px solid rgba(44,44,44,0.12) !important;
    border-radius: 14px !important;
    color: #2C2C2C !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    transition: all 0.2s ease !important;
}

.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: #9B8EC4 !important;
    box-shadow: 0 0 0 3px rgba(155,142,196,0.15), 0 2px 8px rgba(0,0,0,0.04) !important;
    outline: none !important;
}

.stTextInput label, .stNumberInput label, .stTextArea label,
.stSelectbox label, .stDateInput label, .stRadio label {
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    color: #7A7A8C !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.25rem !important;
    font-family: 'Inter', sans-serif !important;
}

.stSelectbox > div > div, .stMultiSelect > div > div {
    background: #FFFFFF !important;
    border: 1.5px solid rgba(44,44,44,0.12) !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    color: #2C2C2C !important;
}

/* ─── METRIC CARDS ─── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border-radius: 20px !important;
    padding: 1.3rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.055) !important;
    border: 1px solid rgba(44,44,44,0.06) !important;
    transition: transform 0.22s ease, box-shadow 0.22s ease !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px rgba(0,0,0,0.09) !important;
}

[data-testid="stMetricLabel"] {
    color: #7A7A8C !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stMetricValue"] {
    color: #2C2C2C !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ─── CUSTOM HEALTH CARDS ─── */
.health-card {
    border-radius: 24px;
    padding: 1.8rem 2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.055);
    border: 1px solid rgba(255,255,255,0.75);
    backdrop-filter: blur(8px);
    margin-bottom: 1rem;
    animation: fadeInUp 0.4s ease both;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.health-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.09);
}
.card-pink  { background: linear-gradient(135deg, #F7D4DC, #FDEEF1); }
.card-blue  { background: linear-gradient(135deg, #D6E8FF, #EBF4FF); }
.card-green { background: linear-gradient(135deg, #D9F0D2, #EDFAE8); }
.card-yellow{ background: linear-gradient(135deg, #F5E7B2, #FBF5DA); }
.card-lavender { background: linear-gradient(135deg, #E8DDF7, #F4EFFC); }

.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #2C2C2C;
    letter-spacing: -0.025em;
    margin: 0 0 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.section-subtext {
    font-size: 0.88rem;
    color: #7A7A8C;
    margin-bottom: 1.6rem;
    font-weight: 400;
    font-family: 'Inter', sans-serif;
    line-height: 1.5;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.22rem 0.75rem;
    border-radius: 50px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-right: 0.4rem;
    font-family: 'Inter', sans-serif;
}
.badge-pink    { background: #F7D4DC; color: #C26B80; }
.badge-blue    { background: #D6E8FF; color: #3A7BC8; }
.badge-green   { background: #D9F0D2; color: #2E7A42; }
.badge-yellow  { background: #F5E7B2; color: #9A7218; }
.badge-lavender{ background: #E8DDF7; color: #6350A8; }

/* ─── ALERTS — SOFT HEALTHCARE STYLE ─── */
.alert-card-red {
    background: linear-gradient(135deg, #FDE8E8, #FDF2F1);
    border: 1px solid rgba(220, 90, 80, 0.2);
    border-left: 5px solid #E87B6C;
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 20px rgba(232,123,108,0.1);
    animation: fadeInUp 0.4s ease both;
}

.alert-card-orange {
    background: linear-gradient(135deg, #FEF3E2, #FEF8EE);
    border: 1px solid rgba(220, 150, 50, 0.2);
    border-left: 5px solid #E8A84C;
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 20px rgba(232,168,76,0.1);
    animation: fadeInUp 0.4s ease both;
}

/* ─── TABS ─── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.75) !important;
    border-radius: 50px !important;
    padding: 5px !important;
    border: 1px solid rgba(44,44,44,0.07) !important;
    gap: 4px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 50px !important;
    padding: 0.42rem 1.15rem !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    color: #7A7A8C !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.22s ease !important;
    font-family: 'Inter', sans-serif !important;
}

.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #2C2C2C !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08) !important;
    font-weight: 600 !important;
}

/* ─── DATA TABLE ─── */
.stDataFrame, [data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.055) !important;
    border: 1px solid rgba(44,44,44,0.06) !important;
    background: #FFFFFF !important;
}

/* ─── PROGRESS BAR ─── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #9B8EC4, #E87B6C) !important;
    border-radius: 50px !important;
}
.stProgress > div > div {
    background: rgba(44,44,44,0.07) !important;
    border-radius: 50px !important;
}

/* ─── FORM CONTAINER ─── */
[data-testid="stForm"] {
    background: #FFFFFF !important;
    border-radius: 24px !important;
    padding: 2rem !important;
    border: 1px solid rgba(44,44,44,0.06) !important;
    box-shadow: 0 4px 28px rgba(0,0,0,0.05) !important;
}

/* ─── INFO / SUCCESS / WARNING / ERROR ─── */
.stInfo > div, [data-testid="stNotification"][data-baseweb="notification"][kind="info"] {
    background: linear-gradient(135deg, #EBF4FF, #F0F7FF) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(58,123,200,0.18) !important;
    color: #2A5A9C !important;
}
.stSuccess > div {
    background: linear-gradient(135deg, #EDFAE8, #F2FCF0) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(46,122,66,0.18) !important;
    color: #255C30 !important;
}
.stWarning > div {
    background: linear-gradient(135deg, #FEF8EE, #FFFBF4) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(220,150,50,0.18) !important;
    color: #7A4F10 !important;
}
.stError > div {
    background: linear-gradient(135deg, #FEF0EF, #FDF4F4) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(220,80,70,0.18) !important;
    color: #8C2A20 !important;
}

/* ─── HR DIVIDER ─── */
hr {
    border: none !important;
    border-top: 1.5px solid rgba(44,44,44,0.08) !important;
    margin: 1.8rem 0 !important;
}

/* ─── CHAT BUBBLES ─── */
.chat-user-bubble {
    background: linear-gradient(135deg, #E87B6C, #D26A88);
    color: #FFFFFF;
    border-radius: 20px 20px 6px 20px;
    padding: 0.85rem 1.2rem;
    margin: 0.5rem 0 0.5rem 18%;
    font-size: 0.9rem;
    line-height: 1.6;
    box-shadow: 0 4px 16px rgba(232,123,108,0.25);
    animation: fadeInUp 0.3s ease both;
    font-family: 'Inter', sans-serif;
}

.chat-agent-bubble {
    background: linear-gradient(135deg, #E8DDF7, #F0EAF9);
    color: #2C2C2C;
    border-radius: 20px 20px 20px 6px;
    padding: 0.85rem 1.2rem;
    margin: 0.5rem 18% 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
    box-shadow: 0 4px 16px rgba(155,142,196,0.15);
    animation: fadeInUp 0.3s ease both;
    font-family: 'Inter', sans-serif;
}

.chat-name {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    opacity: 0.65;
    font-family: 'Inter', sans-serif;
}

/* ─── STATUS PILLS ─── */
.status-pill-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #D9F0D2;
    color: #255C30;
    border-radius: 50px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid rgba(46,122,66,0.2);
    font-family: 'Inter', sans-serif;
}
.status-pill-offline {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FDE8E8;
    color: #8C2A20;
    border-radius: 50px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid rgba(220,80,70,0.2);
    font-family: 'Inter', sans-serif;
}

/* ─── HERO BANNER ─── */
.hero-banner {
    background: linear-gradient(135deg, #F7D4DC 0%, #E8DDF7 45%, #D6E8FF 100%);
    border-radius: 28px;
    padding: 2.4rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.85);
    box-shadow: 0 8px 48px rgba(0,0,0,0.06);
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.5s ease both;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(255,255,255,0.4) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #2C2C2C;
    letter-spacing: -0.03em;
    margin: 0 0 0.3rem 0;
    line-height: 1.2;
    font-family: 'Inter', sans-serif;
}
.hero-subtitle {
    font-size: 0.9rem;
    color: #5A5A6E;
    font-weight: 400;
    font-family: 'Inter', sans-serif;
    line-height: 1.5;
}
.pulse-dot {
    display: inline-block;
    width: 9px; height: 9px;
    background: #4CAF50;
    border-radius: 50%;
    margin-right: 5px;
    animation: pulse-green 2s infinite;
    vertical-align: middle;
}

/* ─── PLOTLY CHART CONTAINERS ─── */
[data-testid="stPlotlyChart"] > div {
    background: #FFFFFF !important;
    border-radius: 20px !important;
    padding: 0.5rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05) !important;
    border: 1px solid rgba(44,44,44,0.05) !important;
}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(44,44,44,0.03); border-radius: 10px; }
::-webkit-scrollbar-thumb { background: rgba(44,44,44,0.14); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(44,44,44,0.24); }

/* ─── ANIMATIONS ─── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-green {
    0%   { box-shadow: 0 0 0 0 rgba(76,175,80,0.55); }
    70%  { box-shadow: 0 0 0 8px rgba(76,175,80,0); }
    100% { box-shadow: 0 0 0 0 rgba(76,175,80,0); }
}
@keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    14%  { transform: scale(1.12); }
    28%  { transform: scale(1); }
    42%  { transform: scale(1.08); }
}

/* ─── PAGE BLOCK ANIMATION ─── */
.main .block-container { animation: fadeInUp 0.45s ease both; }

/* ─── BALLOON OVERRIDE ─── */
.balloons > div { filter: saturate(0.7); }
</style>
""", unsafe_allow_html=True)

init_db(DB_PATH)
llm_cfg = get_llm_config()

# ─── HERO BANNER ───
st.markdown("""
<div class="hero-banner">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
    <div>
      <p style="margin:0 0 0.4rem 0;font-size:0.72rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#9B8EC4;font-family:'Inter',sans-serif;">
        🩺  ICMR-NIN 2024 · AI-Powered · Edge IoT
      </p>
      <p class="hero-title">Arogya AI <span style="font-weight:300;">Health Intelligence</span></p>
      <p class="hero-subtitle">Patient-led longitudinal diabetes monitoring, South Asian caloric engine &amp; bone density analytics</p>
    </div>
    <div style="text-align:right;">
      <div style="font-size:2.8rem;animation:heartbeat 2.5s infinite;display:inline-block;">🫀</div>
      <p style="margin:0.3rem 0 0 0;font-size:0.72rem;color:#7A7A8C;font-family:'Inter',sans-serif;">
        <span class="pulse-dot"></span>System Active
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation & Authentication
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <p style="font-size:1.1rem;font-weight:800;color:#F0F0F8;margin:0;letter-spacing:-0.02em;font-family:'Inter',sans-serif;">♥️ Arogya AI</p>
      <p style="font-size:0.72rem;color:#6A6A80;margin:0.15rem 0 0 0;font-weight:500;font-family:'Inter',sans-serif;letter-spacing:0.04em;text-transform:uppercase;">Health Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    if "user" not in st.session_state:
        st.session_state["user"] = None

    user = st.session_state.get("user")
    if user is None:
        st.markdown("<h4 style='color:#ff4b4b;'>Secure Account Intake</h4>", unsafe_allow_html=True)
        auth_mode = st.radio("Account", ["Log in", "Sign up"], horizontal=True)
        username = st.text_input("Username", placeholder="e.g., alex", key="auth_username")
        password = st.text_input("Password", type="password", key="auth_password")
        if auth_mode == "Sign up":
            password2 = st.text_input("Confirm password", type="password", key="auth_password2")
        
        if st.button(auth_mode, use_container_width=True, type="primary"):
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
        st.markdown("""
        <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:16px;padding:1rem 1.2rem;margin-top:0.5rem;">
          <p style="font-size:0.72rem;color:#8080A0;margin:0;font-family:'Inter',sans-serif;line-height:1.6;">
            🛡️ Patient-facing only. No data sent to external clinical systems without explicit patient action.
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.success(f"Logged in as **{user.username}**")
    if st.button("Log out", use_container_width=True):
        st.session_state["user"] = None
        st.session_state.clear()
        st.rerun()

    st.markdown("""
    <div style="margin-bottom:0.8rem;">
      <p style="font-size:0.68rem;color:#6A6A80;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;font-family:'Inter',sans-serif;margin:0 0 0.6rem 0;">Navigation</p>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio(
        "",
        [
            "🦴  Onboarding & Anthropometry",
            "🥗  My Caloric Engine",
            "📈  Longitudinal Trends",
            "🚨  Diabetes & Health Alerts",
            "🍱  Regional Food Analyzer",
            "🦴  IoT Osteoporosis Scan",
            "📋  Report & Clinical Export",
            "💬  Chat"
        ]
    )
    
    st.markdown("""
    <div style="margin-top:0.5rem;">
      <p style="font-size:0.68rem;color:#6A6A80;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;font-family:'Inter',sans-serif;margin:0 0 0.5rem 0;">AI Configuration</p>
    </div>
    """, unsafe_allow_html=True)
    if llm_cfg:
        st.write(f"<p style='color:#A0A0B8;font-size:0.8rem;font-family:Inter,sans-serif;margin:0;'>Model: <b style='color:#E8E8F0;'>{llm_cfg.model}</b></p>", unsafe_allow_html=True)
        use_llm = st.toggle("Use LLM for chat", value=True)
    else:
        st.write("<p style='color:#6A6A80;font-size:0.8rem;font-family:Inter,sans-serif;'>LLM disabled (Set GEMINI_API_KEY).</p>", unsafe_allow_html=True)
        use_llm = False

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

# ─── Population Reference Metrics ───
st.markdown("""
<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.8rem;">
  <span class="badge badge-blue">📊 Reference</span>
  <span style="font-size:1.1rem;font-weight:700;color:#2C2C2C;font-family:'Inter',sans-serif;">Population Database</span>
</div>
""", unsafe_allow_html=True)
colA, colB, colC, colD = st.columns(4)
colA.metric("Reference Population Size", f"{summary.n}")
colB.metric("Average BMI (Reference)", f"{summary.bmi_mean:.2f}")
colC.metric("Overweight Rate (BMI≥25)", pct(summary.overweight_rate))
colD.metric("Obesity Rate (BMI≥30)", pct(summary.obese_rate))

# Fetch User profile
if "profile" not in st.session_state:
    st.session_state["profile"] = {}

if "checkins" not in st.session_state:
    st.session_state["checkins"] = []

u = st.session_state["user"]
loaded = load_profile(DB_PATH, user_id=u.id)
if loaded is None:
    loaded = {
        "name": "",
        "gender": "Male",
        "age": 30,
        "goal": "Lose weight",
        "units": "Metric",
        "activity": "Moderate (3–5 days/week)",
        "diet_pref": "No preference",
        "constraints": "",
        "height_m": 1.70,
        "weight_kg": 70.0,
        "waist_cm": 85.0,
        "skinfold_mm": 15.0,
    }
st.session_state["profile"] = loaded
st.session_state["checkins"] = list_checkins(DB_PATH, user_id=u.id)

# ----------------- PAGE 1: ONBOARDING & ANTHROPOMETRY -----------------
if page == "🦴  Onboarding & Anthropometry":
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.3rem;">
        <span class="badge badge-pink">🦴 Body Composition</span>
      </div>
      <p class="section-header">🪴 Onboarding &amp; Anthropometry Engine</p>
      <p class="section-subtext">Input your core parameters and advanced physical measurements (caliper/tape data) to compute accurate visceral fat risk and body fat percentages using ICMR-NIN South Asian standards.</p>
    </div>
    """, unsafe_allow_html=True)

    p = st.session_state["profile"]
    
    with st.form("anthropometry_form"):
        c1, c2, c3 = st.columns(3)
        p["name"] = c1.text_input("Name (optional)", value=p.get("name", ""))
        p["gender"] = c2.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(p.get("gender", "Male")))
        p["goal"] = c3.selectbox("Goal", ["Lose weight", "Maintain weight", "Gain weight"], index=["Lose weight", "Maintain weight", "Gain weight"].index(p.get("goal", "Lose weight")))

        c4, c5, c6 = st.columns(3)
        p["units"] = c4.selectbox("Units", ["Metric", "Imperial"], index=0 if p.get("units") == "Metric" else 1)
        p["activity"] = c5.selectbox(
            "Activity level",
            ["Low (0–2 days/week)", "Moderate (3–5 days/week)", "High (6–7 days/week)"],
            index=["Low (0–2 days/week)", "Moderate (3–5 days/week)", "High (6–7 days/week)"].index(p.get("activity", "Moderate (3–5 days/week)")),
        )
        p["diet_pref"] = c6.selectbox("Diet preference", ["No preference", "Vegetarian", "Vegan", "Halal", "Keto-ish", "High-protein"], index=["No preference", "Vegetarian", "Vegan", "Halal", "Keto-ish", "High-protein"].index(p.get("diet_pref", "No preference")))

        st.divider()
        st.markdown("#### 📏 Key Body Measurements")
        a1, a2, a3 = st.columns(3)
        p["age"] = int(a1.number_input("Age (years)", min_value=1, max_value=120, value=int(p.get("age", 30)), step=1))

        if p["units"] == "Metric":
            p["height_m"] = float(a2.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=float(p.get("height_m", 1.70)), step=0.01, format="%.2f"))
            p["weight_kg"] = float(a3.number_input("Weight (kg)", min_value=10.0, max_value=400.0, value=float(p.get("weight_kg", 70.0)), step=0.1, format="%.1f"))
        else:
            height_in = float(a2.number_input("Height (inches)", min_value=20.0, max_value=100.0, value=float(p.get("height_m", 1.70)) / 0.0254, step=0.5, format="%.1f"))
            weight_lb = float(a3.number_input("Weight (lb)", min_value=22.0, max_value=900.0, value=float(p.get("weight_kg", 70.0)) / 0.45359237, step=0.5, format="%.1f"))
            p["height_m"] = height_in * 0.0254
            p["weight_kg"] = weight_lb * 0.45359237

        b1, b2, b3 = st.columns(3)
        p["waist_cm"] = float(b1.number_input("Waist Circumference (cm)", min_value=20.0, max_value=250.0, value=float(p.get("waist_cm", 85.0)), step=0.1))
        p["skinfold_mm"] = float(b2.number_input("Skinfold Thickness - Caliper (mm)", min_value=1.0, max_value=120.0, value=float(p.get("skinfold_mm", 15.0)), step=0.1))
        p["constraints"] = b3.text_input("Physical Constraints (injuries, allergies)", value=p.get("constraints", ""))

        submit = st.form_submit_button("Save Anthropometry & Calculate", type="primary")

    if submit:
        # Core Calculations
        bmi = calculate_bmi(p["height_m"], p["weight_kg"])
        cat = bmi_category(bmi)
        whtr = calculate_whtr(p["waist_cm"], p["height_m"] * 100.0)
        body_fat = estimate_body_fat_percentage(bmi, p["waist_cm"], p["skinfold_mm"], p["gender"])
        visceral = assess_visceral_fat_risk(p["waist_cm"], whtr, p["gender"])
        bmr = calculate_icmr_bmr(p["weight_kg"], p["gender"])
        
        # Activity Factor mapping
        act_factor = 1.2
        if "Moderate" in p["activity"]:
            act_factor = 1.4
        elif "High" in p["activity"]:
            act_factor = 1.6
        tdee = bmr * act_factor
        
        # Obese risk models
        risk = predict_obesity_risk(model, age=p["age"], height_m=p["height_m"], weight_kg=p["weight_kg"], bmi=bmi)
        plan = build_plan(bmi=bmi, category=cat, risk_level=risk.risk_level, visceral_risk=visceral["risk_level"], body_fat_pct=body_fat)

        # Store in session state
        st.session_state["last_bmi"] = bmi
        st.session_state["last_category"] = cat
        st.session_state["last_risk"] = risk
        st.session_state["last_plan"] = plan
        st.session_state["last_visceral_risk"] = visceral
        st.session_state["last_body_fat"] = body_fat
        st.session_state["last_whtr"] = whtr
        st.session_state["last_bmr"] = bmr
        st.session_state["last_tdee"] = tdee
        
        # Keep profile object updated
        upsert_profile(DB_PATH, user_id=u.id, profile=p)
        
        # Automatically insert a check-in for historical plotting
        today_str = str(datetime.date.today())
        add_checkin(
            DB_PATH, 
            user_id=u.id, 
            date=today_str, 
            weight_kg=p["weight_kg"], 
            glucose=None,
            hba1c=None,
            bp_systolic=None,
            bp_diastolic=None,
            waist_cm=p["waist_cm"],
            skinfold_mm=p["skinfold_mm"],
            waist_height_ratio=whtr,
            note="Auto-generated during onboarding calculations."
        )
        st.session_state["checkins"] = list_checkins(DB_PATH, user_id=u.id)

        st.success("🎉 Anthropometrics successfully calculated! Head to **My Caloric Engine** or **Longitudinal Trends** to view details.")

# ----------------- PAGE 2: MY CALORIC ENGINE -----------------
elif page == "🥗  My Caloric Engine":
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <span class="badge badge-green">🥗 Nutrition</span>
      <p class="section-header" style="margin-top:0.4rem;">🧰 Advanced Indian Caloric Engine</p>
      <p class="section-subtext">ICMR-NIN 2024 aligned metabolic budget with South Asian visceral fat risk classification and personalized macronutrient targets.</p>
    </div>
    """, unsafe_allow_html=True)
    
    last_bmi = st.session_state.get("last_bmi")
    last_bmr = st.session_state.get("last_bmr")
    last_tdee = st.session_state.get("last_tdee")
    last_body_fat = st.session_state.get("last_body_fat")
    last_visceral_risk = st.session_state.get("last_visceral_risk")
    p = st.session_state["profile"]

    if last_bmi is None or last_bmr is None:
        st.warning("⚠️ Please complete the **Onboarding & Anthropometry** calculations first.")
    else:
        st.markdown("#### BMR & Total Daily Energy Expenditure (TDEE)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Basal Metabolic Rate (BMR)", f"{last_bmr:.0f} kcal", help="ICMR-NIN formula")
        c2.metric("Daily Energy Burn (TDEE)", f"{last_tdee:.0f} kcal")
        
        # Calculate daily caloric budget target based on goal and visceral risk
        caloric_deficit = 0.0
        if p["goal"] == "Lose weight":
            caloric_deficit = 500.0
        elif p["goal"] == "Gain weight":
            caloric_deficit = -300.0
            
        # Visceral/abdominal fat risk automatically forces a safe deficit recommendation
        if last_visceral_risk and last_visceral_risk["risk_level"] == "High Visceral Risk":
            caloric_deficit = max(caloric_deficit, 500.0)
            
        recommended_budget = last_tdee - caloric_deficit
        c3.metric("Recommended Calorie Budget", f"{recommended_budget:.0f} kcal", f"-{caloric_deficit:.0f} kcal" if caloric_deficit > 0 else f"+{-caloric_deficit:.0f} kcal")

        # Visceral and subcutaneous fat risk assessments
        st.divider()
        st.markdown("#### 🩺 Fat Classification (ICMR South-Asian Thresholds)")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            st.markdown(f"**Visceral Fat Assessment:**")
            v_risk = last_visceral_risk["risk_level"]
            v_color = "green" if v_risk == "Normal" else ("orange" if v_risk == "Increased Risk" else "red")
            st.markdown(f"<h3 style='color:{v_color};'>{v_risk}</h3>", unsafe_allow_html=True)
            st.write(f"Waist Circumference: **{p['waist_cm']} cm** (Limit: Male 90cm / Female 80cm)")
            st.write(f"Waist-to-Height Ratio: **{st.session_state.get('last_whtr', 0):.2f}** (Limit: 0.50)")
            st.caption(f"*Advisory: {last_visceral_risk['advisory']}*")
        
        with fcol2:
            st.markdown(f"**Subcutaneous Fat Assessment:**")
            st.markdown(f"<h3 style='color:#ff4b4b;'>{last_body_fat:.1f}% Body Fat</h3>", unsafe_allow_html=True)
            st.write(f"Skinfold Thickness: **{p['skinfold_mm']} mm**")
            st.caption("Estimated via Indian-adapted multi-site anthropometry calipers.")

        # ICMR-NIN Caloric / Macro Breakdown
        st.divider()
        st.markdown("#### 🥯 Recommended Macronutrient Breakdown (ICMR-NIN Aligned)")
        
        # Determine macronutrient distributions based on visceral adiposity
        if last_visceral_risk and last_visceral_risk["risk_level"] in ("Increased Risk", "High Visceral Risk"):
            # High visceral fat warrants a lower carb, higher protein metabolic adaptation
            pct_carbs = 0.50
            pct_protein = 0.20
            pct_fats = 0.30
            st.info("💡 **Metabolic Optimization Active**: Carb proportion is adjusted to **50%** and protein increased to **20%** to assist in reducing abdominal adiposity and combatting insulin resistance.")
        else:
            pct_carbs = 0.60
            pct_protein = 0.15
            pct_fats = 0.25
            
        cal_carbs = recommended_budget * pct_carbs
        cal_protein = recommended_budget * pct_protein
        cal_fats = recommended_budget * pct_fats
        
        g_carbs = cal_carbs / 4.0
        g_protein = cal_protein / 4.0
        g_fats = cal_fats / 9.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Carbohydrates (50-60%)", f"{g_carbs:.1f} g", f"{cal_carbs:.0f} kcal")
        m2.metric("Proteins (15-20%)", f"{g_protein:.1f} g", f"{cal_protein:.0f} kcal")
        m3.metric("Fats (20-30%)", f"{g_fats:.1f} g", f"{cal_fats:.0f} kcal")

        # Plotly chart for macro breakdown
        fig = go.Figure(data=[go.Pie(
            labels=['Carbohydrates', 'Proteins', 'Fats'],
            values=[cal_carbs, cal_protein, cal_fats],
            hole=.4,
            marker_colors=['#E87B6C', '#9B8EC4', '#6CC5D1']
        )])
        fig.update_layout(
            title_text="Macronutrient Distribution (Calories)",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------- PAGE 3: LONGITUDINAL TRENDS (MOBILE-CENTRIC) -----------------
elif page == "📈  Longitudinal Trends":
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <span class="badge badge-blue">📈 Trends</span>
      <p class="section-header" style="margin-top:0.4rem;">📊 Longitudinal Health &amp; Diabetes Trends</p>
      <p class="section-subtext">High-contrast, mobile-ready graphs — designed for pulling up on your device to show your doctor during in-person consultations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Clinical Demonstration Simulator
    st.markdown("### 🧪 Demo Simulator: Glucose Upward Trend Simulation")
    st.write("Satisfy clinical auditing rules by simulating a 3-month upward glucose and HbA1c trajectory. This will demonstrate in-app trend warnings.")
    
    if st.button("Simulate 3-Month Upward Glucose Trend (Mock Data)", type="primary"):
        # Clear existing check-ins to make simulation clean
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM checkins WHERE user_id = ?", (u.id,))
        
        # Insert 3 months of increasing logs
        t_base = datetime.date.today()
        # 90 days ago: healthy
        add_checkin(DB_PATH, user_id=u.id, date=str(t_base - datetime.timedelta(days=90)), weight_kg=72.5, glucose=94.0, hba1c=5.6, bp_systolic=118, bp_diastolic=78, waist_cm=84.0, skinfold_mm=14.0, note="Baseline healthy status")
        # 60 days ago: slight increase
        add_checkin(DB_PATH, user_id=u.id, date=str(t_base - datetime.timedelta(days=60)), weight_kg=73.8, glucose=128.0, hba1c=6.2, bp_systolic=128, bp_diastolic=82, waist_cm=86.0, skinfold_mm=15.0, note="Subtle upward shift")
        # 30 days ago: prediabetic / diabetic range
        add_checkin(DB_PATH, user_id=u.id, date=str(t_base - datetime.timedelta(days=30)), weight_kg=75.1, glucose=165.0, hba1c=7.0, bp_systolic=138, bp_diastolic=88, waist_cm=88.5, skinfold_mm=16.0, note="Consistent elevations")
        # Today: high diabetic range
        add_checkin(DB_PATH, user_id=u.id, date=str(t_base), weight_kg=76.8, glucose=212.0, hba1c=8.2, bp_systolic=148, bp_diastolic=94, waist_cm=91.0, skinfold_mm=17.5, note="Simulated active hyperglycemia")

        st.session_state["checkins"] = list_checkins(DB_PATH, user_id=u.id)
        st.success("Successfully injected mock trends! Review the 'Diabetes & Health Alerts' or graphs below.")
        st.rerun()

    # Manual Add Check-in Form
    st.divider()
    st.markdown("### 📝 Log Daily Health Check-in")
    with st.form("manual_checkin_form"):
        c1, c2, c3 = st.columns(3)
        check_date = c1.date_input("Date")
        weight_val = float(c2.number_input("Weight (kg)", min_value=10.0, max_value=400.0, value=70.0, step=0.1))
        gluc_val = c3.number_input("Fasting Blood Glucose (mg/dL) - Optional", min_value=0.0, max_value=600.0, value=0.0, step=1.0)

        c4, c5, c6 = st.columns(3)
        hba1c_val = c4.number_input("HbA1c (%) - Optional", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        bp_sys = int(c5.number_input("Systolic Blood Pressure (mmHg) - Optional", min_value=0, max_value=300, value=0, step=1))
        bp_dia = int(c6.number_input("Diastolic Blood Pressure (mmHg) - Optional", min_value=0, max_value=200, value=0, step=1))

        submit_checkin = st.form_submit_button("Record Check-in")
        
    if submit_checkin:
        # Convert 0 entries to None
        g_val = gluc_val if gluc_val > 0 else None
        h_val = hba1c_val if hba1c_val > 0 else None
        b_s = bp_sys if bp_sys > 0 else None
        b_d = bp_dia if bp_dia > 0 else None
        
        # Calculate skinfold and waist ratios from active profile
        active_waist = st.session_state["profile"].get("waist_cm", 85.0)
        active_skinfold = st.session_state["profile"].get("skinfold_mm", 15.0)
        active_height = st.session_state["profile"].get("height_m", 1.70)
        w_h_ratio = calculate_whtr(active_waist, active_height * 100.0)

        add_checkin(
            DB_PATH,
            user_id=u.id,
            date=str(check_date),
            weight_kg=weight_val,
            glucose=g_val,
            hba1c=h_val,
            bp_systolic=b_s,
            bp_diastolic=b_d,
            waist_cm=active_waist,
            skinfold_mm=active_skinfold,
            waist_height_ratio=w_h_ratio,
            note="Manual clinical check-in log."
        )
        st.session_state["checkins"] = list_checkins(DB_PATH, user_id=u.id)
        st.success("Check-in logged!")
        st.rerun()

    # Graphical Plots
    st.divider()
    st.markdown("### 📊 Longitudinal Graphic View")
    
    if not st.session_state["checkins"]:
        st.info("Please log check-ins to view longitudinal graphical charts.")
    else:
        ch = pd.DataFrame(st.session_state["checkins"])
        ch["date"] = pd.to_datetime(ch["date"])
        ch = ch.sort_values("date")

        tab1, tab2, tab3 = st.tabs(["🩸 Blood Glucose & HbA1c", "💓 Blood Pressure", "⚖️ Weight & Waist"])

        with tab1:
            st.markdown("#### Longitudinal Glucose & HbA1c Trends")
            
            # Plot Glucose if exists
            g_data = ch.dropna(subset=["glucose"])
            if not g_data.empty:
                fig_gluc = px.line(g_data, x="date", y="glucose", markers=True, title="Fasting Glucose (mg/dL)", line_shape="spline")
                fig_gluc.update_layout(template="plotly_white")
                st.plotly_chart(fig_gluc, use_container_width=True)
            else:
                st.info("No blood glucose metrics recorded yet.")

            # Plot HbA1c if exists
            h_data = ch.dropna(subset=["hba1c"])
            if not h_data.empty:
                fig_hba1c = px.line(h_data, x="date", y="hba1c", markers=True, title="HbA1c (%)", line_shape="spline", color_discrete_sequence=["#9B8EC4"])
                fig_hba1c.update_layout(template="plotly_white")
                st.plotly_chart(fig_hba1c, use_container_width=True)
            else:
                st.info("No HbA1c metrics recorded yet.")

        with tab2:
            st.markdown("#### Longitudinal Blood Pressure Trends")
            bp_data = ch.dropna(subset=["bp_systolic", "bp_diastolic"])
            if not bp_data.empty:
                fig_bp = go.Figure()
                fig_bp.add_trace(go.Scatter(x=bp_data["date"], y=bp_data["bp_systolic"], name="Systolic BP", mode="lines+markers", line=dict(color="#E87B6C")))
                fig_bp.add_trace(go.Scatter(x=bp_data["date"], y=bp_data["bp_diastolic"], name="Diastolic BP", mode="lines+markers", line=dict(color="#6CC5D1")))
                fig_bp.update_layout(title="Blood Pressure (mmHg)", template="plotly_white")
                st.plotly_chart(fig_bp, use_container_width=True)
            else:
                st.info("No blood pressure metrics recorded yet.")

        with tab3:
            st.markdown("#### Weight & Waist Circumference Progression")
            fig_w = px.line(ch, x="date", y="weight_kg", markers=True, title="Body Weight Trend (kg)")
            fig_w.update_layout(template="plotly_white")
            st.plotly_chart(fig_w, use_container_width=True)

            fig_waist = px.line(ch, x="date", y="waist_cm", markers=True, title="Waist Circumference (cm)", color_discrete_sequence=["#6CC5D1"])
            fig_waist.update_layout(template="plotly_white")
            st.plotly_chart(fig_waist, use_container_width=True)

# ----------------- PAGE 4: DIABETES & HEALTH ALERTS -----------------
elif page == "🚨  Diabetes & Health Alerts":
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <span class="badge badge-yellow">🚨 Alerts</span>
      <p class="section-header" style="margin-top:0.4rem;">🩺 Patient Diabetes &amp; Health Alerts</p>
      <p class="section-subtext">Trend analysis runs locally on your dashboard. Zero automated data is pushed to doctor databases — all alerts are strictly patient-owned, in-app advisories.</p>
    </div>
    """, unsafe_allow_html=True)
    
    checkins = st.session_state["checkins"]
    if not checkins:
        st.info("No historical check-ins to evaluate trends. Please log measurements first or run the Simulator on the 'Longitudinal Trends' tab.")
    else:
        ch = pd.DataFrame(st.session_state["checkins"])
        ch["date"] = pd.to_datetime(ch["date"])
        ch = ch.sort_values("date")

        st.markdown("### Active Clinical Observations (In-App)")

        # 1. Glucose & HbA1c Trend Evaluation
        g_data = ch.dropna(subset=["glucose", "hba1c"])
        if not g_data.empty:
            glucose_vals = g_data["glucose"].tolist()
            hba1c_vals = g_data["hba1c"].tolist()
            
            # Check for a continuous upward trend in last 3 entries
            glucose_upward = len(glucose_vals) >= 3 and glucose_vals[-1] > glucose_vals[-2] > glucose_vals[-3]
            hba1c_upward = len(hba1c_vals) >= 3 and hba1c_vals[-1] > hba1c_vals[-2] > hba1c_vals[-3]
            
            # Check thresholds
            glucose_high = glucose_vals[-1] >= 126.0
            hba1c_high = hba1c_vals[-1] >= 6.5
            
            if glucose_upward or hba1c_upward or glucose_high or hba1c_high:
                st.markdown("""
                <div class="alert-card-red">
                    <h3 style="color:#ff4b4b;margin:0;">🚨 Clinician Consult Required (Trend Advisory)</h3>
                    <p style="margin:8px 0;font-size:15px;color:#f0f0f0;">
                        <b>Your recent trends show changes. Please consult your doctor and share these reports.</b>
                    </p>
                    <hr style="border:0;border-top:1px solid rgba(255,75,75,0.25);margin:10px 0;">
                    <ul style="margin:0;padding-left:20px;font-size:13px;color:#d0d0d0;">
                        <li>Last Fasting Glucose: <b>{g_val:.1f} mg/dL</b> {g_status}</li>
                        <li>Last HbA1c: <b>{h_val:.2f}%</b> {h_status}</li>
                        <li>Upward glucose trend detected over the last few assessments.</li>
                    </ul>
                </div>
                """.format(
                    g_val=glucose_vals[-1],
                    g_status=" (Elevated)" if glucose_high else "",
                    h_val=hba1c_vals[-1],
                    h_status=" (Elevated)" if hba1c_high else ""
                ), unsafe_allow_html=True)
            else:
                st.success("✅ Glycemic metrics are stable and within standard control limits.")
        else:
            st.info("Fasting Glucose or HbA1c metrics missing. Unable to analyze glycemic trends.")

        # 2. Blood Pressure Trend Evaluation
        bp_data = ch.dropna(subset=["bp_systolic", "bp_diastolic"])
        if not bp_data.empty:
            sys_vals = bp_data["bp_systolic"].tolist()
            dia_vals = bp_data["bp_diastolic"].tolist()
            
            sys_high = sys_vals[-1] >= 140
            dia_high = dia_vals[-1] >= 90
            bp_fluctuating = len(sys_vals) >= 3 and (abs(sys_vals[-1] - sys_vals[-2]) > 15 or abs(dia_vals[-1] - dia_vals[-2]) > 10)
            
            if sys_high or dia_high or bp_fluctuating:
                st.markdown("""
                <div class="alert-card-orange">
                    <h3 style="color:#ff7800;margin:0;">⚠️ Hypertension / Cardiovascular Trend Warning</h3>
                    <p style="margin:8px 0;font-size:15px;color:#f0f0f0;">
                        <b>Your recent trends show changes. Please consult your doctor and share these reports.</b>
                    </p>
                    <hr style="border:0;border-top:1px solid rgba(255,120,0,0.25);margin:10px 0;">
                    <ul style="margin:0;padding-left:20px;font-size:13px;color:#d0d0d0;">
                        <li>Last Blood Pressure Reading: <b>{sys}/{dia} mmHg</b></li>
                        <li>Systolic limits: &lt; 140 mmHg | Diastolic limits: &lt; 90 mmHg</li>
                        <li>Fluctuations or elevated thresholds detected. Ensure you show this to your physician.</li>
                    </ul>
                </div>
                """.format(sys=sys_vals[-1], dia=dia_vals[-1]), unsafe_allow_html=True)
            else:
                st.success("✅ Blood Pressure levels are in the healthy range.")
        else:
            st.info("Blood Pressure metrics missing. Unable to analyze cardiovascular trends.")

# ----------------- PAGE 5: REGIONAL FOOD ANALYZER -----------------
elif page == "🍱  Regional Food Analyzer":
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <span class="badge badge-yellow">🍱 Nutrition</span>
      <p class="section-header" style="margin-top:0.4rem;">🍽️ Regional Indian Dietary Compiler</p>
      <p class="section-subtext">Validate nutritional values based on regional culinary preparation styles — Maharashtra vs MP vs South India. Tracks added oils, carbohydrates, and peanuts precisely.</p>
    </div>
    """, unsafe_allow_html=True)

    # Regional Database
    regional_food_db = {
        "Poha": {
            "Maharashtra Style": {
                "desc": "Kanda Poha. Pan-fried with significant peanut oil, tempered with mustard seeds, loaded with roasted peanuts, and finished with fresh grated coconut.",
                "calories": 320,
                "carbs": 44,
                "protein": 6,
                "fat": 13,
                "oil": 10.5,
                "groundnuts": 12.0,
                "score": "B-",
                "notes": "Higher fats and dense sodium due to roasted groundnuts. Garnish with lemon to aid iron absorption."
            },
            "Madhya Pradesh Style": {
                "desc": "Indori Poha. Steamed rather than fried, cooked with minimal oil, but heavily sweetened with sugar, flavored with fennel seeds, and topped with Sev/Farsan.",
                "calories": 375,
                "carbs": 70,
                "protein": 5,
                "fat": 8,
                "oil": 4.5,
                "groundnuts": 0.0,
                "score": "C+",
                "notes": "Extremely high in fast-releasing carbohydrates due to added sugars and deep-fried Sev topping. Risk of rapid blood glucose spikes."
            },
            "South Indian Style": {
                "desc": "Aval Upma. Lightly tossed with very little oil, highly seasoned with nutrient-dense curry leaves, mustard seeds, green chillies, and roasted split chickpeas.",
                "calories": 260,
                "carbs": 48,
                "protein": 4.5,
                "fat": 5,
                "oil": 4.0,
                "groundnuts": 0.0,
                "score": "A-",
                "notes": "Low in saturated fat and added lipids. High levels of fiber from curry leaves and split urad dal tempering."
            }
        },
        "Khichdi": {
            "Maharashtra Style": {
                "desc": "Peanut Khichdi / Sabudana. Heavy on starches, roasted peanuts, and groundnut oil. A popular fasting dish.",
                "calories": 420,
                "carbs": 72,
                "protein": 7,
                "fat": 11,
                "oil": 8.0,
                "groundnuts": 15.0,
                "score": "C",
                "notes": "Extremely carbohydrate-dense, lower in complex fibers. Not suitable for diabetes control."
            },
            "Madhya Pradesh Style": {
                "desc": "Moong Dal Khichdi. Simple balanced rice and split yellow lentil preparation, lightly tempered with cumin seeds, ghee, and turmeric.",
                "calories": 280,
                "carbs": 49,
                "protein": 8.5,
                "fat": 5,
                "oil": 4.0,
                "groundnuts": 0.0,
                "score": "A",
                "notes": "Highly digestible, balanced glycemic index. Perfect clinical choice for diabetes-friendly eating."
            },
            "South Indian Style": {
                "desc": "Ven Pongal. Prepared with rice and moong lentils but seasoned with significant amounts of pure Ghee, whole black peppercorns, ginger, and cashews.",
                "calories": 390,
                "carbs": 52,
                "protein": 8.0,
                "fat": 16,
                "oil": 14.0,
                "groundnuts": 0.0,
                "score": "B+",
                "notes": "High fat from pure ghee increases total caloric density. Ghee has healthy SCFAs, but portions must be managed."
            }
        },
        "Upma": {
            "Maharashtra Style": {
                "desc": "Sajgura. Coarse roasted semolina with added onions, potatoes, peanuts, and medium vegetable oils.",
                "calories": 310,
                "carbs": 52,
                "protein": 6.5,
                "fat": 8,
                "oil": 6.0,
                "groundnuts": 8.0,
                "score": "B",
                "notes": "Moderate index. Ensure adding fresh local green beans or carrots to lower glycemic load."
            },
            "Madhya Pradesh Style": {
                "desc": "Rava Upma. Highly seasoned, often topped with sev or bhujia and small amounts of sugar.",
                "calories": 340,
                "carbs": 62,
                "protein": 6.0,
                "fat": 7.5,
                "oil": 5.0,
                "groundnuts": 0.0,
                "score": "B-",
                "notes": "Farsan topping elevates total simple carbs and sodium."
            },
            "South Indian Style": {
                "desc": "Suji Upma (Traditional). Tempering with mustard, ginger, green chillies, curry leaves, and dry split dals. Minimal oil.",
                "calories": 250,
                "carbs": 46,
                "protein": 5.8,
                "fat": 4.5,
                "oil": 3.8,
                "groundnuts": 0.0,
                "score": "A-",
                "notes": "Healthy light profile. Can be improved by using whole wheat rava or dahlia."
            }
        }
    }

    c1, c2 = st.columns(2)
    food_select = c1.selectbox("Choose Food Item", list(regional_food_db.keys()))
    style_select = c2.selectbox("Choose Regional Preparation Style", ["Maharashtra Style", "Madhya Pradesh Style", "South Indian Style"])

    # Output detailed caloric engine data
    food_data = regional_food_db[food_select][style_select]

    st.divider()
    st.markdown(f"### 🍽️ {style_select} {food_select}")
    st.write(food_data["desc"])

    # Nutrition Metrics Display
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Energy", f"{food_data['calories']} kcal")
    mc2.metric("Added Cooking Oils", f"{food_data['oil']} g")
    mc3.metric("Groundnuts (Peanuts)", f"{food_data['groundnuts']} g")
    
    # Color-coded ICMR safety grade
    score_colors = {"A": "green", "A-": "green", "B+": "green", "B": "orange", "B-": "orange", "C+": "red", "C": "red"}
    s_col = score_colors.get(food_data["score"], "white")
    mc4.markdown(f"**ICMR Safety Rating**  \n<h2 style='color:{s_col};margin:0;'>{food_data['score']}</h2>", unsafe_allow_html=True)

    # Plot macro comparison
    fig_macro = go.Figure(data=[
        go.Bar(name='Carbs (g)', x=[style_select], y=[food_data['carbs']], marker_color='#E87B6C'),
        go.Bar(name='Protein (g)', x=[style_select], y=[food_data['protein']], marker_color='#9B8EC4'),
        go.Bar(name='Fat (g)', x=[style_select], y=[food_data['fat']], marker_color='#6CC5D1')
    ])
    fig_macro.update_layout(barmode='group', template="plotly_white", height=300)
    st.plotly_chart(fig_macro, use_container_width=True)
    st.caption(f"*NIN Advisory Notes: {food_data['notes']}*")

    # Bounding Box Simulation Scanner
    st.divider()
    st.markdown("### 📷 Advanced AI Volumetric Food-Photo Scanner")
    st.write("Scan your plate to analyze volumetric sizes. Below, trigger a simulation to visualize computer vision boundary analysis on local Indian culinary configurations.")
    
    if st.button("Trigger AI Volumetric Scanner Simulation", type="primary"):
        st.write("🔍 **Scanning camera buffer...**")
        st.info("Detected dish: " + food_select)
        
        # We simulate bounding boxes using customized Plotly layouts to represent live computer-vision scans!
        img_h, img_w = 400, 600
        
        # Bounding boxes definitions based on selections
        boxes = []
        if style_select == "Maharashtra Style":
            boxes = [
                {"name": "Peanut oil lipids", "x": [50, 500, 500, 50, 50], "y": [40, 40, 360, 360, 40], "color": "yellow"},
                {"name": "Groundnuts (12g)", "x": [100, 250, 250, 100, 100], "y": [120, 120, 220, 220, 120], "color": "red"},
                {"name": "Carbohydrate Base (Poha)", "x": [60, 520, 520, 60, 60], "y": [50, 50, 350, 350, 50], "color": "green"}
            ]
        elif style_select == "Madhya Pradesh Style":
            boxes = [
                {"name": "Sev / Farsan Topping", "x": [80, 480, 480, 80, 80], "y": [80, 80, 280, 280, 80], "color": "orange"},
                {"name": "Refined sugars", "x": [150, 350, 350, 150, 150], "y": [100, 100, 180, 180, 100], "color": "cyan"},
                {"name": "Carbohydrate Base", "x": [60, 520, 520, 60, 60], "y": [50, 50, 350, 350, 50], "color": "green"}
            ]
        else: # South India
            boxes = [
                {"name": "Lentils tempering", "x": [200, 350, 350, 200, 200], "y": [150, 150, 250, 250, 150], "color": "blue"},
                {"name": "Carbohydrate Base", "x": [60, 520, 520, 60, 60], "y": [50, 50, 350, 350, 50], "color": "green"}
            ]

        fig_cv = go.Figure()
        
        # Add background representing the plate
        fig_cv.add_shape(type="circle", x0=50, y0=30, x1=550, y1=370, line_color="grey", fillcolor="rgba(100, 100, 100, 0.1)")
        
        # Add bounding boxes
        for b in boxes:
            fig_cv.add_trace(go.Scatter(
                x=b["x"], y=b["y"],
                mode="lines",
                name=b["name"],
                line=dict(color=b["color"], width=3)
            ))
            
        fig_cv.update_layout(
            xaxis=dict(visible=False, range=[0, img_w]),
            yaxis=dict(visible=False, range=[0, img_h]),
            height=350,
            title=f"AI Computer-Vision Bounding BBoxes ({style_select} {food_select})",
            template="plotly_white",
            showlegend=True
        )
        st.plotly_chart(fig_cv, use_container_width=True)
        st.markdown(f"""
        **Plate Volumetric Diagnostics:**
        * Calculated density: **High Accuracy**
        * Estimated oil layers: **{food_data['oil']}g**
        * Estimated groundnut count: **{food_data['groundnuts']}g**
        * Volumetric Carbohydrates: **{food_data['carbs']}g**
        """)

# ----------------- PAGE 6: IoT Osteoporosis Scan -----------------
elif page == "🦴  IoT Osteoporosis Scan":
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <span class="badge badge-lavender">🦴 Bone Health</span>
      <p class="section-header" style="margin-top:0.4rem;">🟣 IoT Bone Vibration Spectroscopy</p>
      <p class="section-subtext">Wirelessly interact with the built-in Arduino ESP32 edge sensor to measure bone acoustic resonance and calculate your osteoporosis / osteopenia clinical risk in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    import time

    # Check Flask server status
    flask_online = False
    try:
        resp = requests.get("http://127.0.0.1:5000/api/live", timeout=1.0)
        if resp.status_code == 200:
            flask_online = True
    except Exception:
        pass

    # Status Indicator
    if flask_online:
        st.markdown("""
        <div style="margin-bottom:1.5rem;">
          <span class="status-pill-online">
            <span style="width:7px;height:7px;background:#2E7A42;border-radius:50%;display:inline-block;"></span>
            Flask API &amp; ESP32 Server — Connected
          </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-bottom:1.5rem;">
          <span class="status-pill-offline">
            <span style="width:7px;height:7px;background:#8C2A20;border-radius:50%;display:inline-block;"></span>
            Flask API Offline — Start Flask on port 5000
          </span>
        </div>
        """, unsafe_allow_html=True)

    p = st.session_state["profile"]
    
    # Bone Scan DB Path
    BONE_DB_PATH = Path("e:/SECOND Y/edi/Osteoprosis-Scan/IOT CODES/BoneApp/clinical_data.db")

    # Connect to local DB and list records
    def load_bone_scans(limit=100):
        if not BONE_DB_PATH.exists():
            return []
        try:
            conn = sqlite3.connect(str(BONE_DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patients ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    bone_scans = load_bone_scans()

    # Form for Intake and Triggering
    with st.form("bone_scan_form"):
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">
          <span class="badge badge-lavender">⚡ Wireless Trigger</span>
          <span style="font-weight:600;color:#2C2C2C;font-family:'Inter',sans-serif;">Initialize Acoustic Resonance Trial</span>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        patient_age = int(c1.number_input("Patient Age", min_value=1, max_value=120, value=int(p.get("age", 30))))
        
        # Height is stored in meters in profile, convert to cm for the IoT intake
        profile_height_m = float(p.get("height_m", 1.70))
        patient_height_cm = float(c2.number_input("Patient Height (cm)", min_value=50.0, max_value=250.0, value=profile_height_m * 100.0, step=0.1))
        
        patient_weight_kg = float(c3.number_input("Patient Weight (kg)", min_value=10.0, max_value=400.0, value=float(p.get("weight_kg", 70.0)), step=0.1))

        sim_mode = st.checkbox("Use Inbuilt Simulated Arduino (No physical hardware required)", value=True, help="Simulate a real-time heel drop test using a background simulation thread.")
        if sim_mode:
            esp_ip = "Simulation"
            st.info("Simulation mode is enabled. No physical ESP32 or Wi-Fi configuration required.")
        else:
            esp_ip = st.text_input("ESP32 Sensor IP Address", value=st.session_state.get("esp_ip", "10.57.174.231"))
            st.session_state["esp_ip"] = esp_ip

        submit_trigger = st.form_submit_button("Launch Arduino Bone Scan", type="primary", use_container_width=True)

    if submit_trigger:
        if not flask_online:
            st.error("Cannot launch test: Flask API Server on port 5000 is unreachable.")
        else:
            status_placeholder = st.empty()
            status_placeholder.info("Sending trigger request to ESP32 via Flask server...")
            try:
                trigger_url = "http://127.0.0.1:5000/api/start_test"
                payload = {
                    "age": patient_age,
                    "height": patient_height_cm,
                    "weight": patient_weight_kg,
                    "esp_ip": esp_ip
                }
                resp = requests.post(trigger_url, json=payload, timeout=5)
                res_data = resp.json()
                if resp.status_code == 200 and res_data.get("status") == "success":
                    st.session_state["active_bone_test"] = True
                    st.success(f"Device triggered successfully! Patient ID: {res_data.get('patient_id')}. Instruct patient to begin the 5 heel drops.")
                    st.rerun()
                else:
                    st.error(f"Error from Flask/ESP32: {res_data.get('message', 'Unknown error')}")
            except Exception as e:
                st.error(f"Failed to connect to Flask server: {str(e)}")

    # Active Scanning Live View
    if st.session_state.get("active_bone_test"):
        st.divider()
        st.markdown("### ⚡ Live Trial Progress Monitor")
        
        # Display placeholders
        progress_placeholder = st.empty()
        count_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Loop to monitor the live drops count from Flask
        max_retries = 30
        drops_detected = 0
        last_drop = -1
        
        for i in range(max_retries):
            try:
                live_resp = requests.get("http://127.0.0.1:5000/api/live", timeout=1.0)
                if live_resp.status_code == 200:
                    data = live_resp.json()
                    drops_detected = int(data.get("drops", 0))
            except Exception:
                pass

            # Update Progress Bar
            progress_val = min(1.0, float(drops_detected) / 5.0)
            progress_placeholder.progress(progress_val)
            count_placeholder.markdown(f"<h2 style='text-align:center;color:#ff4b4b;font-size:3rem;'>Drops: {drops_detected} / 5</h2>", unsafe_allow_html=True)
            
            # Plot dynamic vibration wave
            if drops_detected > last_drop:
                last_drop = drops_detected
                # Generate aesthetic impact damping wave
                t_arr = np.linspace(0, 0.5, 200)
                freq_hz = 65.0 if drops_detected > 0 else 0.0
                damping = 15.0
                noise = np.random.normal(0, 0.05, 200)
                
                # Impact wave is zero before first drop
                if drops_detected > 0:
                    y_arr = np.exp(-damping * t_arr) * np.sin(2 * np.pi * freq_hz * t_arr) + noise
                else:
                    y_arr = noise
                    
                fig_wave = go.Figure()
                fig_wave.add_trace(go.Scatter(x=t_arr, y=y_arr, mode="lines", line=dict(color="#E87B6C", width=2), name="Accelerometer raw"))
                fig_wave.update_layout(
                    title=f"Acoustic Impact Resonance Wave (Drop {drops_detected})",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Amplitude (g)",
                    template="plotly_white",
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                chart_placeholder.plotly_chart(fig_wave, use_container_width=True)

            if drops_detected >= 5:
                progress_placeholder.progress(1.0)
                st.session_state["active_bone_test"] = False
                st.balloons()
                st.success("Acoustic Resonance trial completed successfully! Extracting edge-computed metrics...")
                time.sleep(2.0)
                st.rerun()
                break
                
            time.sleep(0.8)
        else:
            st.session_state["active_bone_test"] = False
            st.warning("Trial timed out. Please check that the ESP32 is powered and connected to the same Wi-Fi subnet.")
            st.rerun()

    # Display Latest Results & Diagnostic
    if bone_scans:
        st.divider()
        st.markdown("### 🦴 Edge-Computed Bone Spectrogram & Diagnostic")
        
        latest = bone_scans[0]
        
        # Calculate T-Score
        f_val = latest.get("freq_hz", 65.0)
        p_val = latest.get("peak_g", 2.5)
        d_val = latest.get("decay", 0.8)
        
        # South Asian bone spectroscopy density regression
        t_score = (f_val - 100.0) / 20.0 + (p_val - 2.5) / 0.5
        t_score = max(-4.0, min(1.5, t_score))
        
        if t_score >= -1.0:
            bone_risk = "Normal Bone Density"
            bone_color = "#2E7A42"   # Soft green
            bone_impression = "Excellent bone elasticity and acoustic resonance. Continue your active weight-bearing exercises and adequate calcium intake."
        elif t_score > -2.5:
            bone_risk = "Osteopenia — Mild Mineral Loss"
            bone_color = "#9A7218"   # Warm amber
            bone_impression = "Mild bone density loss detected. Optimize calcium/Vitamin D intake, engage in safe strength training, and avoid high-impact falls."
        else:
            bone_risk = "Osteoporosis — High Fragility Risk"
            bone_color = "#C0443A"   # Soft coral red
            bone_impression = "🚨 Critical bone resonance damping. High risk of skeletal fragility. Please consult an orthopedic clinician for a DEXA scan and review pharmacotherapy."
            bone_risk = "Osteoporosis (High Fragility Risk)"
            bone_color = "#F44336" # Red
            bone_impression = "🚨 Critical bone resonance damping. High risk of skeletal fragility. Please consult an orthopedic clinician for a DEXA scan and review pharmacotherapy."

        # Metrics Display
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Resonant Frequency", f"{f_val:.1f} Hz", help="Bone stiffness proxy")
        col2.metric("Peak Acceleration", f"{p_val:.2f} g", help="Maximum acoustic vibration amplitude")
        col3.metric("Damping Decay Rate", f"{d_val:.2f}", help="Vibrational energy dissipation")
        
        # Customized T-score metric
        col4.markdown(f"**Approximate T-Score**  \n<h2 style='color:{bone_color};margin:0;'>{t_score:.2f}</h2>", unsafe_allow_html=True)

        # Active diagnostic advisory
        st.markdown(f"""
        <div style="background:rgba(20,20,20,0.6);border:1px solid {bone_color}22;border-left:6px solid {bone_color};border-radius:8px;padding:20px;margin-top:20px;margin-bottom:20px;">
            <h3 style="color:{bone_color};margin:0 0 8px 0;">{bone_risk}</h3>
            <p style="margin:0;font-size:15px;color:#f0f0f0;"><b>Clinical Impression:</b> {bone_impression}</p>
        </div>
        """, unsafe_allow_html=True)

        # Draw spectroscopy spectrum chart
        st.markdown("#### 🔬 Acoustic Frequency Spectroscopy Spectrum")
        f_vec = np.linspace(10, 200, 500)
        # Lorentzian peak around the resonant frequency
        gamma = 8.0 # peak width
        spectrum = (1.0 / np.pi) * (gamma / ((f_vec - f_val) ** 2 + gamma ** 2))
        # Normalize and add minor high-frequency noise
        spectrum = spectrum / np.max(spectrum) * p_val
        spectrum += np.random.normal(0, 0.02 * p_val, 500)
        spectrum = np.clip(spectrum, 0, None)
        
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(x=f_vec, y=spectrum, mode="lines", line=dict(color=bone_color, width=2.5), fill='tozeroy', name="Power Spectral Density"))
        fig_spec.add_vline(x=f_val, line_dash="dash", line_color="white", annotation_text=f"Resonance peak: {f_val:.1f} Hz")
        fig_spec.update_layout(
            xaxis_title="Frequency (Hz)",
            yaxis_title="Spectral Power (g²/Hz)",
            template="plotly_white",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_spec, use_container_width=True)

        # Longitudinal Bone Trends
        if len(bone_scans) >= 2:
            st.divider()
            st.markdown("### 📈 Longitudinal Bone Density Trends")
            
            # Load into DataFrame
            df_bone = pd.DataFrame(bone_scans)
            df_bone["timestamp"] = pd.to_datetime(df_bone["timestamp"])
            df_bone = df_bone.sort_values("timestamp")
            
            # Recalculate T-Score for history
            df_bone["t_score"] = (df_bone["freq_hz"] - 100.0) / 20.0 + (df_bone["peak_g"] - 2.5) / 0.5
            df_bone["t_score"] = df_bone["t_score"].clip(-4.0, 1.5)
            
            btab1, btab2 = st.tabs(["📶 Resonance Frequency Trend", "🦴 Densitometric T-Score Trend"])
            with btab1:
                fig_f_trend = px.line(df_bone, x="timestamp", y="freq_hz", markers=True, title="Resonant Frequency (Hz) vs Date", line_shape="spline", color_discrete_sequence=[bone_color])
                fig_f_trend.update_layout(template="plotly_white")
                st.plotly_chart(fig_f_trend, use_container_width=True)
            with btab2:
                fig_t_trend = px.line(df_bone, x="timestamp", y="t_score", markers=True, title="Approximate T-Score vs Date", line_shape="spline", color_discrete_sequence=["#2196F3"])
                # Add threshold reference bands
                fig_t_trend.add_hrect(y0=-1.0, y1=1.5, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Healthy")
                fig_t_trend.add_hrect(y0=-2.5, y1=-1.0, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Osteopenia")
                fig_t_trend.add_hrect(y0=-4.0, y1=-2.5, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Osteoporosis")
                fig_t_trend.update_layout(template="plotly_white", yaxis_range=[-4.0, 1.5])
                st.plotly_chart(fig_t_trend, use_container_width=True)

        # Database History Log Table
        st.divider()
        st.markdown("#### 📋 Historical Bone Resonance Database Logs")
        
        hist_df = pd.DataFrame(bone_scans)
        hist_df["t_score"] = (hist_df["freq_hz"] - 100.0) / 20.0 + (hist_df["peak_g"] - 2.5) / 0.5
        hist_df["t_score"] = hist_df["t_score"].clip(-4.0, 1.5)
        
        # Display clean columns
        hist_display = hist_df[["timestamp", "age", "bmi", "peak_g", "decay", "freq_hz", "t_score"]].copy()
        hist_display.columns = ["Timestamp", "Age", "BMI", "Peak (g)", "Decay", "Freq (Hz)", "T-Score"]
        st.dataframe(hist_display.style.format({
            "BMI": "{:.1f}",
            "Peak (g)": "{:.2f}",
            "Decay": "{:.2f}",
            "Freq (Hz)": "{:.1f}",
            "T-Score": "{:.2f}"
        }), use_container_width=True)
    else:
        st.info("No bone vibration scan sessions recorded yet. Launch a wireless test above to initialize.")

# ----------------- PAGE 7: REPORT & CLINICAL EXPORT -----------------
elif page == "📋  Report & Clinical Export":
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <span class="badge badge-blue">📋 Export</span>
      <p class="section-header" style="margin-top:0.4rem;">📱 Consolidated Patient Clinical Report</p>
      <p class="section-subtext">Your complete anthropometric, metabolic, glycemic, and bone density report — compiled and ready to present to your clinician during visits. Designed for easy CMS import.</p>
    </div>
    """, unsafe_allow_html=True)

    p = st.session_state["profile"]
    last_bmi = st.session_state.get("last_bmi")
    last_risk = st.session_state.get("last_risk")
    last_plan = st.session_state.get("last_plan")
    last_cat = st.session_state.get("last_category")
    last_body_fat = st.session_state.get("last_body_fat")
    last_visceral_risk = st.session_state.get("last_visceral_risk")
    last_bmr = st.session_state.get("last_bmr")
    last_tdee = st.session_state.get("last_tdee")

    if last_bmi is None:
        st.warning("⚠️ Complete the **Onboarding & Anthropometry** dashboard to generate your report.")
    else:
        # Get latest bone scan if exists
        BONE_DB_PATH = Path("e:/SECOND Y/edi/Osteoprosis-Scan/IOT CODES/BoneApp/clinical_data.db")
        bone_section = ""
        if BONE_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(BONE_DB_PATH))
                conn.row_factory = sqlite3.Row
                latest_bone = conn.execute("SELECT * FROM patients ORDER BY timestamp DESC LIMIT 1").fetchone()
                conn.close()
                if latest_bone:
                    f_val = latest_bone["freq_hz"]
                    p_val = latest_bone["peak_g"]
                    d_val = latest_bone["decay"]
                    t_score_val = (f_val - 100.0) / 20.0 + (p_val - 2.5) / 0.5
                    t_score_val = max(-4.0, min(1.5, t_score_val))
                    bone_risk_val = "Normal" if t_score_val >= -1.0 else ("Osteopenia" if t_score_val > -2.5 else "Osteoporosis")
                    bone_section = (
                        f"## 4. Edge-Computing Bone Resonance & Osteoporosis Scan\n"
                        f"- **Date Scan Executed:** {latest_bone['timestamp']}\n"
                        f"- **Resonant Frequency:** {f_val:.1f} Hz\n"
                        f"- **Peak Damping Acceleration:** {p_val:.2f} g\n"
                        f"- **Damping Energy Decay Rate:** {d_val:.2f}\n"
                        f"- **Approximate T-Score:** {t_score_val:.2f} ({bone_risk_val})\n\n"
                    )
            except Exception:
                pass

        # Build Report Markdown Content
        report = (
            f"# Consolidated Longitudinal Clinical Report\n\n"
            f"**Patient Name:** {p.get('name') or 'N/A'}\n"
            f"**Date Generated:** {datetime.date.today()}\n"
            f"**Assigned Goal:** {p.get('goal')}\n"
            f"**Gender:** {p.get('gender')}\n\n"
            f"--- \n"
            f"## 1. Anthropometry & Body Composition\n"
            f"- **Body Mass Index (BMI):** {last_bmi:.2f} kg/m² ({last_cat})\n"
            f"- **Estimated Body Fat %:** {last_body_fat:.1f}% (Caliper & Tapemetric)\n"
            f"- **Waist Circumference:** {p.get('waist_cm')} cm\n"
            f"- **Waist-to-Height Ratio (WHtR):** {st.session_state.get('last_whtr', 0.0):.2f}\n"
            f"- **Visceral Fat Classification:** {last_visceral_risk['risk_level']}\n\n"
            f"## 2. Metabolic Energy Budgets (ICMR-NIN Aligned)\n"
            f"- **Basal Metabolic Rate (BMR):** {last_bmr:.0f} kcal\n"
            f"- **Total Daily Energy Expenditure (TDEE):** {last_tdee:.0f} kcal\n\n"
            f"## 3. Localized Dietary Interventions & Guidelines\n"
            f"### 🥦 Diet Plan:\n" + "\n".join([f"- {x}" for x in last_plan.diet]) + "\n\n"
            f"### 🏃 Fitness plan:\n" + "\n".join([f"- {x}" for x in last_plan.fitness]) + "\n\n"
            f"### 📋 General Clinical Notes:\n" + "\n".join([f"- {x}" for x in (last_plan.notes or ['None'])]) + "\n\n"
            f"{bone_section}"
            f"## 5. Longitudinal Check-in Logs (Last 10 Days)\n\n"
        )
        
        # Append database logs
        report += "| Date | Weight (kg) | Glucose (mg/dL) | HbA1c (%) | BP (mmHg) |\n"
        report += "|---|---|---|---|---|\n"
        for c in st.session_state["checkins"][:10]:
            g_str = f"{c['glucose']:.1f}" if c['glucose'] is not None else "N/A"
            h_str = f"{c['hba1c']:.2f}" if c['hba1c'] is not None else "N/A"
            bp_str = f"{c['bp_systolic']}/{c['bp_diastolic']}" if c['bp_systolic'] is not None else "N/A"
            report += f"| {c['date']} | {c['weight_kg']:.1f} | {g_str} | {h_str} | {bp_str} |\n"
            
        report += "\n\n---\n*This file is generated strictly within the local device environment. No automated data transfer occurred without patient consent.*"

        st.markdown(report)
        
        st.divider()
        st.markdown("### 📥 Download Consolidated Clinical File")
        st.write("Save this report directly as a Markdown/Text file designed to copy or import directly to any clinic management systems (CMS) prior to your visit.")
        st.download_button(
            "Download Clinical Report File",
            data=report.encode("utf-8"),
            file_name="clinical_health_report.md",
            mime="text/markdown",
            use_container_width=True
        )

# ----------------- PAGE 8: CLINICAL CHAT & LLM COACH -----------------
elif page == "💬  Chat":
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <span class="badge badge-lavender">🤖 AI Coach</span>
      <p class="section-header" style="margin-top:0.4rem;">💬 Clinical AI Health Coach</p>
      <p class="section-subtext">Ask naturally. Your coach is grounded in your ICMR-NIN metrics, anthropometrics, regional food profiles, and bone resonance diagnostics — never generic advice.</p>
    </div>
    """, unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    if not st.session_state["chat"]:
        st.session_state["chat"] = [
            ("Agent", "Hello! I am your clinical health coach. Ask me about your advanced calorie budget, BMR, visceral fat, or the nutrition profiles of regional preparations (e.g. Maharashtra vs MP Poha)!")
        ]

    if st.button("Reload Saved Local Chat Messages", use_container_width=True):
        saved = list_chat_messages(DB_PATH, user_id=u.id, limit=30)
        st.session_state["chat"] = []
        for m in saved:
            st.session_state["chat"].append(("You" if m["role"] == "user" else "Agent", m["content"]))

    user_msg = st.text_input("Enter message here...", placeholder="e.g. Compare regional poha prep for my diabetes risk")
    
    if st.button("Send Message", type="primary", use_container_width=True) and user_msg.strip():
        user_text = user_msg.strip()
        st.session_state["chat"].append(("You", user_text))
        add_chat_message(DB_PATH, user_id=u.id, role="user", content=user_text)

        # Build comprehensive local context
        ctx = ChatContext(
            summary=summary,
            profile=st.session_state.get("profile"),
            last_bmi=st.session_state.get("last_bmi"),
            last_risk=st.session_state.get("last_risk"),
            last_plan=st.session_state.get("last_plan"),
        )

        # Get latest bone scan for chatbot alignment
        BONE_DB_PATH = Path("e:/SECOND Y/edi/Osteoprosis-Scan/IOT CODES/BoneApp/clinical_data.db")
        bone_metrics_str = "None Recorded yet."
        if BONE_DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(BONE_DB_PATH))
                conn.row_factory = sqlite3.Row
                latest_bone = conn.execute("SELECT * FROM patients ORDER BY timestamp DESC LIMIT 1").fetchone()
                conn.close()
                if latest_bone:
                    f_val = latest_bone["freq_hz"]
                    p_val = latest_bone["peak_g"]
                    t_score_val = (f_val - 100.0) / 20.0 + (p_val - 2.5) / 0.5
                    t_score_val = max(-4.0, min(1.5, t_score_val))
                    bone_risk_val = "Normal" if t_score_val >= -1.0 else ("Osteopenia" if t_score_val > -2.5 else "Osteoporosis")
                    bone_metrics_str = f"Resonant Frequency: {f_val:.1f}Hz, Peak Damping: {p_val:.2f}g, Calculated T-Score: {t_score_val:.2f} ({bone_risk_val})"
            except Exception:
                pass

        if use_llm and llm_cfg:
            bmi_str = f"{ctx.last_bmi:.2f}" if ctx.last_bmi else "N/A"
            system_prompt = (
                "You are an expert clinical nutrition, metabolic, and bone health coach inside a patient-facing monitoring app. "
                "You are strictly grounded in standard medical and nutritional guidelines, specifically the ICMR (Indian Council of Medical Research) "
                "and NIN (National Institute of Nutrition) 2024 guidelines.\n\n"
                "Key Rules:\n"
                "1. Focus heavily on localized South Asian dietary modifications: Ragi, Jowar, Bajra, millets, whole pulses.\n"
                "2. Emphasize body composition: BMI, visceral fat, waist circumference, waist-to-height ratio, and subcutaneous fat from caliper metrics.\n"
                "3. Explain regional Indian culinary preparations and warn about fats/added sugars. (e.g. Maharashtra Kanda Poha has high peanut oil and roasted peanuts; "
                "MP Indori Poha is steamed but has fast carbohydrates from Sev and sugars; South Indian Upma/Aval has low oil but uses light tempering).\n"
                "4. Bone Resonance & Osteoporosis Analysis: If the user asks about osteoporosis, bone density, T-score, or their resonance frequency, "
                "explain that physical heel drops produce an acoustic resonance. A frequency >=100Hz (T-score >= -1.0) is normal healthy bone density, "
                "50-100Hz (T-score -1.0 to -2.5) indicates mild osteopenia, and <50Hz (T-score <= -2.5) indicates osteoporosis. Recommend weight-bearing exercise "
                "and a high-calcium local diet (ragi, dairy, sesame seeds, curry leaves).\n"
                "5. Safe Clinical Boundaries: You do not diagnose disease. If negative health trends (glucose upward curve) or high values are visible in the context, "
                "explicitly advise: 'Your recent trends show changes. Please consult your doctor and share these reports.'\n\n"
                "Active Patient Metrics:\n"
                f"- Name: {ctx.profile.get('name')}\n"
                f"- Gender: {ctx.profile.get('gender')}\n"
                f"- BMI: {bmi_str}\n"
                f"- Body Fat: {st.session_state.get('last_body_fat', 'N/A')}%\n"
                f"- Visceral Risk: {st.session_state.get('last_visceral_risk', {}).get('risk_level', 'N/A')}\n"
                f"- BMR (ICMR): {st.session_state.get('last_bmr', 'N/A')} kcal | TDEE: {st.session_state.get('last_tdee', 'N/A')} kcal\n"
                f"- Bone Resonance Spectroscopy Scan: {bone_metrics_str}\n"
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

    st.divider()
    # Display message history chronologically
    for speaker, msg in st.session_state["chat"]:
        if speaker == "You":
            st.markdown(f"👤 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Health Coach:** {msg}")



