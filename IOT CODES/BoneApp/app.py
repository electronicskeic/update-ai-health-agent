from flask import Flask, request, jsonify, render_template, Response
import sqlite3
import os
import csv
import io
import requests

app = Flask(__name__)
DB_FILE = "clinical_data.db"

# --- LIVE TRACKING VARIABLE ---
live_drop_count = 0

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    age INTEGER,
                    bmi REAL,
                    peak_g REAL,
                    std_peak REAL,
                    decay REAL,
                    std_decay REAL,
                    freq_hz REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

init_db()

# --- WEB UI ROUTE ---
@app.route('/')
def dashboard():
    """Serves the beautiful HTML dashboard."""
    return render_template('index.html')

# --- DATA FETCH API (For the UI) ---
@app.route('/api/data', methods=['GET'])
def get_data():
    """Sends database data to the web interface."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM patients ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify(rows)

# --- LIVE PING API (From ESP32) ---
@app.route('/ping', methods=['GET'])
def ping():
    """Receives a micro-ping from the ESP32 each time a drop is finished."""
    global live_drop_count
    live_drop_count = int(request.args.get('drop', 0))
    return "OK", 200

# --- LIVE STATUS API (For Website) ---
@app.route('/api/live', methods=['GET'])
def live_status():
    """Sends the current drop count to the website banner."""
    global live_drop_count
    return jsonify({"drops": live_drop_count})

# --- CSV DOWNLOAD API (For ML Training) ---
@app.route('/api/download_csv', methods=['GET'])
def download_csv():
    """Exports the SQLite database to a CSV file for Machine Learning."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM patients ORDER BY timestamp DESC")
    rows = c.fetchall()
    
    # Get column headers
    column_names = [description[0] for description in c.description]
    conn.close()
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(column_names)  # Write headers
    writer.writerows(rows)         # Write data
    
    # Return as a downloadable file
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=clinical_data.csv"}
    )

# --- ESP32 TRIGGER API (Patient Intake) ---
@app.route('/api/start_test', methods=['POST'])
def start_test():
    """Calculates BMI, gets next ID, and triggers the ESP32 wirelessly."""
    global live_drop_count
    
    data = request.json
    age = data.get('age')
    height_cm = float(data.get('height'))
    weight_kg = float(data.get('weight'))
    esp_ip = data.get('esp_ip', '10.57.174.231') # Default from your script
    
    # Reset drop count for the new patient test
    live_drop_count = 0
    
    # Calculate BMI
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 2)
    
    # Get Next Patient ID from SQLite Database (Instead of CSV)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT MAX(id) FROM patients")
    max_id = c.fetchone()[0]
    conn.close()
    
    patient_id = (max_id or 0) + 1
    # Get the laptop IP dynamically from the request
    laptop_ip = request.host.split(':')[0]
    
    # Send the start command to ESP32
    url = f"http://{esp_ip}/start?id={patient_id}&age={age}&bmi={bmi}&server={laptop_ip}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return jsonify({
                "status": "success",
                "message": f"Device triggered. Instruct patient to begin heel drops.",
                "patient_id": patient_id,
                "bmi": bmi
            })
        else:
            return jsonify({"status": "error", "message": f"ESP32 error: {response.status_code}"}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"status": "error", "message": f"Connection failed: {str(e)}. Check ESP32 IP."}), 500

# --- ESP32 UPLOAD API (The Receiver) ---
@app.route('/upload', methods=['POST'])
def receive_data():
    """Catches JSON from the ESP32 and saves it to SQLite."""
    global live_drop_count
    data = request.json
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''INSERT INTO patients 
                 (patient_id, age, bmi, peak_g, std_peak, decay, std_decay, freq_hz)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
              (data['id'], data['age'], data['bmi'], data['peak'], 
               data['stdPeak'], data['decay'], data['stdDecay'], data['freq']))
    conn.commit()
    conn.close()
    
    print(f"✅ Saved Patient {data['id']} to Database!")
    live_drop_count = 0 # Reset counter for the next patient
    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    # Runs the server on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)
