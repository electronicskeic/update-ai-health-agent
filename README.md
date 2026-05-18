# Osteoporosis-Scan: Edge-Computing Clinical Dashboard & AI Health Agent

**Osteoporosis-Scan** is a complete, end-to-end medical IoT and AI-driven platform. It is designed to monitor and analyze acoustic bone resonance to assess osteoporosis and osteopenia risks. This repository is unique because it comes with the **hardware (Arduino/ESP32) component built-in**, natively integrating with a modern Flask-based clinical dashboard and a Streamlit-powered AI Health Agent.

## ✨ Key Features & "Built-in Arduino" Integration

- **Built-in Arduino (ESP32) Codebase:** The project includes native C++ code for the ESP32 (`esp32_sensor.ino`), eliminating the need for third-party bridges. The ESP32 acts as an edge-computing node, directly sampling data from an **MPU6050** accelerometer via I2C, running its own HTTP web server, and handling two-way communication with the clinical dashboard.
- **Real-Time Data Streaming:** The built-in Arduino firmware exposes a `/start` endpoint to trigger testing. It actively streams live progress pings (`/ping`) during heel-drop monitoring and autonomously uploads the final calculated metrics (Peak Acceleration, Decay, Frequency) via a JSON POST request (`/upload`).
- **Clinical Dashboard (Flask):** A sleek, dark-themed, glassmorphic UI that allows clinicians to control the ESP32 remotely, track live test progress, manage patient intake forms (Age, BMI), and save clinical data to a local SQLite database.
- **AI Health Agent (Streamlit & Gemini):** An AI-driven companion that analyzes patient BMI, correlates age with BMI using dataset insights, generates tailored diet/fitness plans, and includes an interactive clinical chatbot.
- **Data Export:** Easily export clinical session data from the dashboard to CSV for machine learning or statistical analysis.

---

## 📂 Project Structure

```text
Osteoprosis-Scan/
│
├── IOT CODES/
│   ├── Arduino_Code/
│   │   └── esp32_sensor/
│   │       └── esp32_sensor.ino      # Built-in C++ code for the ESP32 Microcontroller
│   └── BoneApp/
│       ├── app.py                    # Flask Web Server & API endpoints
│       ├── clinical_data.db          # SQLite Database for patient data
│       └── templates/
│           └── index.html            # The Red/Dark Minimalist Dashboard UI
│
└── aihealthagent/
    └── ai-health-agent-main/
        ├── app.py                    # Streamlit AI Health Agent
        ├── requirements.txt          # Python dependencies for the AI Agent
        └── health_agent/             # AI core logic, chatbot, and data models
```

---

## 🚀 Getting Started

### 1. Hardware Setup (Built-in ESP32 Firmware)
1. Open `IOT CODES/Arduino_Code/esp32_sensor/esp32_sensor.ino` in the Arduino IDE.
2. Update your Wi-Fi credentials:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. Connect an **MPU6050** accelerometer to the ESP32 via I2C pins.
4. Upload the sketch to the ESP32 board. Open the Serial Monitor (115200 baud) to find the ESP32's IP address.

### 2. Running the Clinical Dashboard
The main dashboard serves as the central hub, communicating directly with the built-in Arduino code.
1. Open a terminal and navigate to the Flask app folder:
   ```bash
   cd "IOT CODES/BoneApp"
   ```
2. Install Python dependencies:
   ```bash
   pip install flask requests
   ```
3. Run the dashboard:
   ```bash
   python app.py
   ```
4. Open your browser and navigate to `http://localhost:5000`.

### 3. Running the AI Health Agent
The AI Health Agent is a separate service that analyzes the generated metrics.
1. Open a **second** terminal window and navigate to the AI agent folder:
   ```bash
   cd "aihealthagent/ai-health-agent-main"
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. From the main dashboard (`http://localhost:5000`), you can click the **Launch AI Health Agent** button in the top right to seamlessly open the AI dashboard (`http://localhost:8501`).

---

## ⚙️ How It Works (The Workflow)

1. **Intake:** The clinician enters the patient's Age, Height, Weight, and the ESP32's IP address on the Flask dashboard.
2. **Triggering:** Clicking "Initialize Test" sends an HTTP GET request (`/start`) to the built-in ESP32 web server.
3. **Tracking:** The patient performs heel drops. The ESP32 detects the impacts via the MPU6050 and sends live progress pings (`/ping`) back to the Flask server to update the dashboard banner.
4. **Data Upload:** Once finished, the ESP32 calculates metrics (Peak G, Decay, Frequency) and pushes a JSON payload back to the Flask server (`/upload`).
5. **AI Analysis:** The clinician clicks "Launch AI Health Agent" to cross-reference the patient's stats, get dataset-driven risk insights, and interact with the Gemini AI for tailored health recommendations.

---

## ⚠️ Disclaimer
**Educational Purpose Only.** This project is not intended to provide professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider regarding medical conditions.