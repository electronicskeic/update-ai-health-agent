<div align="center">

<img src="https://img.shields.io/badge/Arogya%20AI-Health%20Intelligence-E87B6C?style=for-the-badge&logo=heart&logoColor=white" alt="Arogya AI"/>
<img src="https://img.shields.io/badge/ICMR--NIN%202024-Aligned-9B8EC4?style=for-the-badge" alt="ICMR NIN"/>
<img src="https://img.shields.io/badge/Arduino%20ESP32-IoT%20Integration-6CC5D1?style=for-the-badge&logo=arduino&logoColor=white" alt="ESP32"/>
<img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>

# 🩺 Arogya AI — Patient Health Intelligence Platform

### *A premium, patient-led longitudinal health monitoring system powered by South Asian clinical standards, edge-computing IoT bone spectroscopy, and AI-driven nutritional analysis.*

[Features](#-features) · [Screenshots](#-design) · [Installation](#-installation) · [IoT Setup](#-iot--esp32-setup) · [Architecture](#-architecture) · [API Reference](#-api-reference) · [Formulas](#-clinical-formulas)

</div>

---

## 📌 Overview

**Arogya AI** is a full-stack, patient-centric health intelligence platform built for Indian patients managing diabetes, metabolic syndrome, and bone health. It combines:

- 📊 **Longitudinal diabetes & cardiovascular trend monitoring**
- 🍽️ **ICMR-NIN 2024 aligned South Asian caloric engine**
- 🦴 **Edge-computing Arduino/ESP32 bone vibration spectroscopy (osteoporosis scanner)**
- 🤖 **Gemini-powered clinical AI health coach**
- 📱 **Mobile-ready clinical report export for in-person doctor consultations**

> **Philosophy:** Zero automated alerts to external clinical systems. Every observation is patient-owned, in-app, and presented in a calming, emotionally supportive interface designed for real patient empowerment.

---

## ✨ Features

### 1. 🪴 Onboarding & Anthropometry Engine

The foundation page. Collects all essential body measurements and computes a complete South Asian-specific body composition profile.

**What it does:**
- Accepts **Age, Gender, Height, Weight** in Metric or Imperial units
- Collects **Waist Circumference (cm)** via tape measure
- Accepts **Skinfold Thickness (mm)** via caliper measurement
- Records physical constraints, diet preferences, and health goals

**Calculated Outputs:**
| Metric | Method |
|---|---|
| BMI | `weight_kg / height_m²` |
| BMI Category | WHO/ICMR South Asian classification |
| Waist-to-Height Ratio (WHtR) | `waist_cm / height_cm` |
| Body Fat % | Multi-site South Asian anthropometric regression |
| Visceral Fat Risk | ICMR waist circumference + WHtR dual-threshold model |
| Basal Metabolic Rate (BMR) | ICMR-NIN 2024 Indian population formula |
| Total Daily Energy Expenditure (TDEE) | BMR × Activity Factor |
| Obesity Risk Score | ML Random Forest trained on 15,000-patient BMI dataset |

**Why ICMR-NIN standards matter:**
> South Asians develop metabolic disease at significantly lower BMI thresholds than Western populations. The ICMR classifies overweight at BMI ≥ 23.0 (vs. 25.0 globally) and sets abdominal obesity thresholds at 90 cm (men) and 80 cm (women) — this application strictly enforces these boundaries.

---

### 2. 🧰 Advanced Indian Caloric Engine

Computes a fully personalised calorie budget and macronutrient breakdown aligned with ICMR-NIN 2024 Dietary Guidelines.

**Features:**
- Displays **BMR**, **TDEE**, and **Recommended Daily Calorie Budget** in floating metric cards
- Automatically adjusts caloric deficit based on goal (Lose / Maintain / Gain)
- **Visceral fat risk override:** If high abdominal adiposity is detected, the engine automatically activates a metabolic intervention protocol

**Metabolic Optimisation (Abdominal Adiposity Protocol):**
```
Standard Profile:    60% Carbs | 15% Protein | 25% Fats
Abdominal Adiposity: 50% Carbs | 20% Protein | 30% Fats  ← auto-activated
```

**Visceral & Subcutaneous Fat Classification:**
- Visceral: Waist circumference + WHtR dual-flag system
- Subcutaneous: South Asian-adapted caliper regression model

**Visual Output:**
- Interactive Plotly donut chart showing calorie distribution by macronutrient
- Grams and kilocalorie values for Carbs, Proteins, and Fats
- ICMR advisory notes on optimal food sources

---

### 3. 📈 Longitudinal Health & Diabetes Trends

A mobile-first clinical dashboard showing health trajectory over time — designed to be shown to a doctor on a phone screen during a clinic visit.

**Tracking Metrics:**
| Metric | Clinical Threshold |
|---|---|
| Fasting Blood Glucose (mg/dL) | Alert if ≥ 126 mg/dL |
| HbA1c (%) | Alert if ≥ 6.5% |
| Systolic Blood Pressure (mmHg) | Alert if ≥ 140 mmHg |
| Diastolic Blood Pressure (mmHg) | Alert if ≥ 90 mmHg |
| Body Weight (kg) | Progressive trend tracking |
| Waist Circumference (cm) | Visceral fat regression |

**Features:**
- **3-Month Mock Trend Simulator** — injects a realistic upward glucose trajectory (94 → 128 → 165 → 212 mg/dL) with matching HbA1c and BP escalation to demonstrate in-app alert triggering
- **Manual daily check-in form** with optional metrics
- **Tabbed spline charts** for Glucose/HbA1c, Blood Pressure, and Weight/Waist
- All charts use smooth `line_shape="spline"` curves for visual clarity
- Every chart auto-stores to SQLite for longitudinal persistence

---

### 4. 🩺 Patient Diabetes & Health Alerts

Evaluates stored longitudinal data and generates in-app clinical advisories. Zero external push notifications — patient-owned observations only.

**Alert Logic:**
```python
# Upward trend (3 consecutive readings)
glucose_upward = glucose[-1] > glucose[-2] > glucose[-3]
hba1c_upward   = hba1c[-1] > hba1c[-2] > hba1c[-3]

# Threshold breaches
glucose_high = glucose[-1] >= 126.0  # mg/dL (IDF diabetic threshold)
hba1c_high   = hba1c[-1] >= 6.5     # % (ADA HbA1c criterion)
sys_high     = systolic[-1] >= 140   # mmHg (Stage 2 Hypertension)
```

**Alert Types:**
| Condition | Visual Style | Message |
|---|---|---|
| Glucose upward trend / high | Soft coral card | "Clinician Consult Required — Trend Advisory" |
| BP elevated / fluctuating | Warm amber card | "Hypertension / Cardiovascular Trend Warning" |
| All metrics stable | Mint success card | "Glycemic metrics are stable and within control limits" |

> **Medical-Legal Compliance:** By design, no alerts are sent via SMS, email, or push notification to any healthcare provider. This prevents liability if a provider misses an alert. Patients are responsible for sharing their in-app report with their doctor.

---

### 5. 🍽️ Regional Indian Dietary Compiler

A detailed nutritional analysis engine that accounts for the significant caloric differences between regional Indian preparations of the same dish.

**Supported Dishes & Regions:**
| Dish | Maharashtra Style | Madhya Pradesh Style | South Indian Style |
|---|---|---|---|
| **Poha** | Kanda Poha (peanut oil + roasted groundnuts) | Indori Poha (steamed, sev topping, sugar) | Aval Upma (light sesame, curry leaves) |
| **Khichdi** | Sabudana (starchy, high peanuts) | Moong Dal Khichdi (balanced) | Ven Pongal (ghee + cashews) |
| **Upma** | Sajgura (semolina, potatoes, peanuts) | Rava Upma (sev topping) | Suji Upma (minimal oil, dal tempering) |

**Nutritional Data Per Preparation:**
- Total Calories (kcal)
- Added Cooking Oil (g) — critical for abdominal fat risk
- Groundnut/Peanut content (g)
- Carbohydrate, Protein, Fat macros (g)
- ICMR Safety Grade (A / A- / B+ / B / B- / C+ / C)
- Clinical advisory notes

**AI Volumetric Food Photo Scanner (Simulation):**
- Simulates a computer-vision bounding-box analysis of a food plate
- Renders Plotly-based bounding box overlays showing lipid layers, carb base, and groundnut regions
- Provides volumetric diagnostics (oil density, carbohydrate mass estimation)

---

### 6. 🦴 IoT Bone Vibration Spectroscopy Scanner

The most technically advanced feature: a real-time, wireless bone density assessment system using an Arduino ESP32 edge-computing accelerometer node, integrated directly into the Streamlit dashboard.

**Hardware Required:**
- ESP32 microcontroller with MPU-6050 or ADXL345 accelerometer
- Wi-Fi network shared by ESP32 and the laptop/server
- Physical heel-drop protocol (5 standardised drops)

**How It Works:**

```
Patient stands → ESP32 triggered wirelessly via Flask API
→ Patient performs 5 heel drops
→ Accelerometer captures each impact wave  
→ ESP32 computes: peak_g, decay rate, resonant frequency
→ Data POSTed to Flask backend → stored in clinical_data.db
→ Streamlit polls live status and renders real-time wave graph
→ T-Score calculated from spectroscopic regression formula
→ Clinical diagnostic displayed with color-coded risk assessment
```

**Live Polling Flow:**
```
Streamlit ──POST /api/start_test──► Flask (port 5000) ──GET /start?id=&bmi=──► ESP32
                                         ◄──GET /ping?drop=N──
Streamlit ──GET /api/live──► Flask (returns current drop count)
                                         ◄──POST /upload {json}──  ESP32 (after all 5 drops)
Streamlit ──GET /api/data──► Flask (final spectroscopy results)
```

**T-Score Regression Formula (South Asian Bone Spectroscopy):**
```
T-Score = clamp(
  (freq_hz - 100.0) / 20.0 + (peak_g - 2.5) / 0.5,
  min = -4.0,
  max = 1.5
)
```

**Clinical Classifications:**
| T-Score Range | Diagnosis | Color | Action |
|---|---|---|---|
| ≥ -1.0 | ✅ Normal Bone Density | Deep Green | Weight-bearing exercise, adequate calcium |
| -1.0 to -2.5 | ⚠️ Osteopenia (Mild Loss) | Warm Amber | Calcium + Vit D, safe strength training |
| ≤ -2.5 | 🚨 Osteoporosis (High Fragility) | Soft Coral Red | Urgent DEXA scan referral |

**Resonant Frequency Bone Correlations:**
| Frequency Range | Bone Density Interpretation |
|---|---|
| ≥ 100 Hz | Rigid, healthy bone matrix |
| 50–100 Hz | Early mineral density loss (Osteopenic) |
| < 50 Hz | Highly porous / fragile bone architecture |

**Visualisations:**
- Real-time acoustic damping wave chart (Plotly) during scanning
- Lorentzian Power Spectral Density spectrum centered at resonant frequency
- Longitudinal T-Score trend graph with color-coded reference bands
- Historical database log table

---

### 7. 📱 Consolidated Clinical Report Export

Compiles all patient data into a single, copy-paste ready clinical document formatted for easy import into clinic management systems.

**Report Sections:**
1. Patient Demographics & Goals
2. Anthropometry & Body Composition (BMI, Body Fat %, WHtR, Visceral Risk)
3. Metabolic Energy Budgets (BMR, TDEE — ICMR formulas)
4. Personalized Diet Plan (ICMR-NIN aligned)
5. Fitness Recommendations
6. Edge-Computing Bone Resonance Scan Results (if available)
7. Longitudinal Check-in Log Table (last 10 entries)

**Export Options:**
- Rendered inline preview (Markdown)
- **One-click Download** as `.md` file for CMS import

---

### 8. 💬 Clinical AI Health Coach

An intelligent chat assistant grounded in the patient's actual clinical metrics — not generic health advice.

**Backend Options:**
| Mode | Description |
|---|---|
| **Gemini LLM** (default) | Uses Google Gemini API with a comprehensive system prompt incorporating all patient metrics |
| **Rule-Based Fallback** | Local Python logic answers clinical questions about BMI, risk, bone density, and nutrition even without an API key |

**System Prompt Context Injected Per Query:**
- Patient name, gender, BMI, body fat %, visceral risk level
- ICMR BMR and TDEE values
- Latest bone resonance scan: frequency, peak_g, T-score, diagnosis
- All ICMR-NIN dietary guidelines, regional food profiles, South Asian metabolic notes

**Topics the Coach Handles:**
- BMI interpretation and South Asian thresholds
- Visceral fat risk and metabolic syndrome
- Regional Indian food preparation comparison (e.g., "Is Maharashtra Poha better than MP Indori Poha for diabetes?")
- Bone density, osteoporosis, T-score interpretation
- Exercise recommendations for South Asian metabolic syndrome
- ICMR-NIN caloric targets and macronutrient guidance

---

## 🎨 Design

The interface is designed as a **premium, calming, futuristic healthcare experience** — inspired by Apple Health, MediKit, and modern hospital SaaS dashboards.

### Design System

| Token | Value | Usage |
|---|---|---|
| Main Background | `#F6F3EC` | Warm cream — reduces eye strain |
| Sidebar | `#111111` | Matte black with `28px` radius |
| Card Pink | `#F7D4DC` | Anthropometry sections |
| Card Blue | `#D6E8FF` | Analytics & trends |
| Card Green | `#D9F0D2` | Nutrition & success states |
| Card Yellow | `#F5E7B2` | Alerts & food analyzer |
| Card Lavender | `#E8DDF7` | Bone health & AI chat |
| Accent Coral | `#E87B6C` | Primary buttons, user bubbles |
| Accent Cyan | `#6CC5D1` | Download buttons |
| Accent Purple | `#9B8EC4` | Chart series, AI elements |
| Text Primary | `#2C2C2C` | Dark charcoal |
| Text Secondary | `#7A7A8C` | Labels, metadata |
| Font | Inter (Google Fonts) | 300–800 weight range |

### Animations
- `fadeInUp` — cards animate on page load
- `heartbeat` — hero icon pulses like a living EKG
- `pulse-green` — live system active indicator
- Hover: `translateY(-2px)` + shadow lift on all cards and buttons

---

## 🏗️ Architecture

```
Osteoprosis-Scan/
│
├── 📁 aihealthagent/ai-health-agent-main/    ← Streamlit Frontend
│   ├── app.py                                 ← Main app (8 pages, CSS, routing)
│   ├── health_agent.db                        ← Patient data SQLite
│   ├── bmi.xlsx                               ← 15,000 patient BMI reference dataset
│   ├── diabetes.csv                           ← Pima Indians diabetes ML dataset
│   ├── requirements.txt
│   ├── .streamlit/
│   │   ├── config.toml                        ← Theme: warm cream + Inter font
│   │   └── secrets.toml                       ← GEMINI_API_KEY
│   └── health_agent/
│       ├── __init__.py
│       ├── storage.py        ← SQLite schema, CRUD (users, checkins, chat)
│       ├── data.py           ← ICMR formulas, anthropometry, dataset loaders
│       ├── model.py          ← Random Forest obesity + diabetes ML models
│       ├── recommendations.py← ICMR-NIN diet & fitness plan builder
│       ├── chatbot.py        ← Local rule-based clinical Q&A engine
│       └── llm.py            ← Gemini API wrapper
│
├── 📁 IOT CODES/
│   ├── 📁 Arduino_Code/                       ← ESP32 Firmware
│   │   └── esp32_sensor.ino                  ← Accelerometer data capture
│   └── 📁 BoneApp/                           ← Flask Microservice (port 5000)
│       ├── app.py                             ← REST API for ESP32 ↔ Streamlit
│       ├── clinical_data.db                  ← Bone scan SQLite database
│       └── templates/
│           └── index.html                    ← Web dashboard for direct Flask access
│
└── README.md
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Patient Browser                                   │
│              http://localhost:8501 (Streamlit)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │      Streamlit Frontend      │
              │   8 Pages + Premium CSS      │
              │   Plotly Charts (light theme)│
              │   Chat Bubbles + Hero Banner │
              └──────────┬──────┬───────────┘
                         │      │
              ┌──────────▼──┐  ┌▼─────────────────────┐
              │ health_agent│  │  Flask Microservice   │
              │    .db      │  │  http://localhost:5000│
              │  (SQLite)   │  │  /api/start_test      │
              │  users      │  │  /api/live (polling)  │
              │  checkins   │  │  /upload (ESP32 POST) │
              │  chat_msgs  │  │  /ping (drop count)   │
              └─────────────┘  └──────────┬────────────┘
                                          │
                              ┌───────────▼───────────┐
                              │   clinical_data.db    │
                              │   (Bone Scan SQLite)  │
                              └───────────┬───────────┘
                                          │ Wi-Fi
                              ┌───────────▼───────────┐
                              │  Arduino ESP32 Node   │
                              │  MPU-6050 / ADXL345   │
                              │  Heel Drop Protocol   │
                              └───────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- pip
- Git
- (Optional) Arduino IDE for ESP32 firmware upload

### 1. Clone the Repository
```bash
git clone https://github.com/electronicskeic/update-ai-health-agent.git
cd update-ai-health-agent
```

### 2. Set Up Virtual Environment (Streamlit App)
```bash
cd aihealthagent/ai-health-agent-main
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
streamlit>=1.28
plotly>=5.18
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
openpyxl>=3.1
requests>=2.31
google-generativeai>=0.4
```

### 4. Configure Gemini API Key (Optional but Recommended)
Edit `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-google-gemini-api-key-here"
```

> Get a free API key at: https://aistudio.google.com/

### 5. Run the Streamlit App
```bash
streamlit run app.py --server.port 8501
```
Open: **http://localhost:8501**

---

## 🔬 IoT / ESP32 Setup

### Hardware
| Component | Purpose |
|---|---|
| ESP32 Dev Board | Wi-Fi + processing |
| MPU-6050 or ADXL345 | 3-axis accelerometer |
| Jumper wires | I2C connection to ESP32 |

### Step 1 — Upload ESP32 Firmware
1. Open `IOT CODES/Arduino_Code/esp32_sensor.ino` in Arduino IDE
2. Install required libraries: `WiFi.h`, `HTTPClient.h`, `Wire.h`, `Adafruit_MPU6050.h`
3. Update your Wi-Fi credentials in the sketch
4. Upload to ESP32

### Step 2 — Start Flask Backend
```bash
cd "IOT CODES/BoneApp"
python -m venv .venv
.venv\Scripts\activate
pip install flask requests

python app.py
# Running on http://0.0.0.0:5000
```

### Step 3 — Run the Full System
1. Start Flask server (port 5000)
2. Start Streamlit app (port 8501)
3. Navigate to **IoT Osteoporosis Scan** page
4. Enter the ESP32's local IP address (find it in Arduino Serial Monitor)
5. Click **"Launch Arduino Bone Scan"**
6. Have the patient perform 5 standardised heel drops
7. Review the Lorentzian spectrum and T-score diagnostic

### Flask API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web dashboard (HTML) |
| `GET` | `/api/data` | Fetch all patient bone scan records |
| `GET` | `/api/live` | Real-time drop count (for polling) |
| `GET` | `/ping?drop=N` | Called by ESP32 after each heel drop |
| `POST` | `/api/start_test` | Trigger ESP32 scan (from Streamlit) |
| `POST` | `/upload` | Receive final spectroscopy JSON from ESP32 |
| `GET` | `/api/download_csv` | Export all data as CSV for ML training |

### `/api/start_test` Payload
```json
{
  "age": 35,
  "height": 170.0,
  "weight": 72.0,
  "esp_ip": "192.168.1.105"
}
```

### ESP32 → Flask `/upload` Payload
```json
{
  "id": 42,
  "age": 35,
  "bmi": 24.91,
  "peak": 3.2,
  "stdPeak": 0.18,
  "decay": 0.75,
  "stdDecay": 0.04,
  "freq": 87.5
}
```

---

## 📐 Clinical Formulas

### ICMR-NIN 2024 Basal Metabolic Rate (BMR)
```
Men:   BMR = (13.5 × Weight_kg) + 487  kcal/day
Women: BMR = (11.5 × Weight_kg) + 483  kcal/day
```
*Source: ICMR-NIN Dietary Guidelines for Indians, 2024*

### Waist-to-Height Ratio (WHtR)
```
WHtR = Waist_cm / Height_cm

Risk Classification:
  WHtR < 0.50  →  Healthy
  WHtR ≥ 0.50  →  Increased Visceral Fat Risk
```

### South Asian Visceral Fat Risk
```
Waist Cutoffs (ICMR):
  Men:    > 90 cm  = High Risk
  Women:  > 80 cm  = High Risk

Dual-flag system: Both WHtR AND waist circumference assessed
  0 flags → Normal
  1 flag  → Increased Risk
  2 flags → High Visceral Risk (metabolic protocol activated)
```

### South Asian Body Fat % Regression
```
Male:   Fat% = (0.45 × BMI) + (0.35 × Waist_cm) + (0.15 × Skinfold_mm) - 10
Female: Fat% = (0.45 × BMI) + (0.35 × Waist_cm) + (0.15 × Skinfold_mm) - 5
```

### Bone Spectroscopy T-Score
```
T-Score = clamp(
  [(freq_hz - 100.0) / 20.0] + [(peak_g - 2.5) / 0.5],
  min = -4.0,
  max = +1.5
)

Clinical Zones:
  T ≥ -1.0             →  Normal (WHO: Healthy)
  -2.5 < T < -1.0      →  Osteopenia (WHO: Mild Loss)
  T ≤ -2.5             →  Osteoporosis (WHO: High Risk)
```

### Lorentzian Spectral Density (Spectrum Chart)
```
S(f) = (1/π) × [γ / ((f - f₀)² + γ²)]

Where:
  f₀ = resonant frequency (Hz)
  γ  = 8.0 Hz (peak width / damping coefficient)

Normalised and scaled by peak_g, then clipped to non-negative values.
```

---

## 🗄️ Database Schema

### `health_agent.db` (Streamlit App)

**`users`**
```sql
CREATE TABLE users (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  username      TEXT UNIQUE NOT NULL,
  password_hash BLOB NOT NULL,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
```

**`checkins`** (longitudinal metrics)
```sql
CREATE TABLE checkins (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id            INTEGER NOT NULL REFERENCES users(id),
  date               TEXT NOT NULL,
  weight_kg          REAL,
  glucose            REAL,         -- mg/dL
  hba1c              REAL,         -- %
  bp_systolic        INTEGER,      -- mmHg
  bp_diastolic       INTEGER,      -- mmHg
  waist_cm           REAL,         -- cm
  skinfold_mm        REAL,         -- mm
  waist_height_ratio REAL,
  note               TEXT,
  created_at         TEXT DEFAULT CURRENT_TIMESTAMP
);
```

**`chat_messages`**
```sql
CREATE TABLE chat_messages (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id    INTEGER NOT NULL REFERENCES users(id),
  role       TEXT NOT NULL,   -- 'user' or 'assistant'
  content    TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### `clinical_data.db` (Flask/IoT Backend)

**`patients`**
```sql
CREATE TABLE patients (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id TEXT,
  age        INTEGER,
  bmi        REAL,
  peak_g     REAL,    -- Max acceleration (g)
  std_peak   REAL,    -- Std deviation of peak
  decay      REAL,    -- Damping decay rate
  std_decay  REAL,    -- Std deviation of decay
  freq_hz    REAL,    -- Resonant frequency (Hz)
  timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🛡️ Medical-Legal Disclaimer

> This application is a **patient education and self-monitoring tool only**. It is not a medical device, does not provide diagnosis, and is not a substitute for professional medical advice.
>
> - The T-score approximation from heel-drop spectroscopy is **not equivalent** to a clinical DEXA scan
> - Blood glucose, HbA1c, and blood pressure trends are **indicative only**
> - All alerts prompt patients to consult a licensed healthcare provider — no automated clinical notifications are sent
> - The ICMR-NIN formulas and South Asian thresholds are referenced from published 2024 guidelines but must be interpreted by a qualified professional

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

**Built with ❤️ for Indian patients by the Arogya AI team**

*Powered by ICMR-NIN 2024 · Arduino ESP32 · Google Gemini · Streamlit · Plotly*

</div>