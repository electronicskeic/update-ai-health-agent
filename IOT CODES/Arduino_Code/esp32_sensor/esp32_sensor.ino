#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ==========================================
// WIFI SETTINGS
// ==========================================
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

WebServer server(80);
Adafruit_MPU6050 mpu;

// ==========================================
// VARIABLES FROM FLASK APP
// ==========================================
String laptop_ip = "";
int patient_id = 0;
int age = 0;
float bmi = 0.0;

bool is_testing = false;
int drop_count = 0;
const int max_drops = 5;

void setup() {
  Serial.begin(115200);
  
  // 1. Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("\nConnected to WiFi!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  // 2. Initialize MPU6050 Accelerometer
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    // Halt if not found (uncomment in production)
    // while (1) { delay(10); }
  } else {
    Serial.println("MPU6050 Found!");
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  }

  // 3. Setup HTTP Server Route for '/start'
  server.on("/start", handleStart);
  server.begin();
  Serial.println("HTTP server started and listening for start commands.");
}

void loop() {
  // Handle incoming HTTP requests from the Laptop
  server.handleClient();

  // If a test was triggered by the laptop, execute the drop tracking
  if (is_testing) {
    performTest();
  }
}

// Handler for the '/start' endpoint hit by the Flask app
void handleStart() {
  if (server.hasArg("id") && server.hasArg("age") && server.hasArg("bmi") && server.hasArg("server")) {
    patient_id = server.arg("id").toInt();
    age = server.arg("age").toInt();
    bmi = server.arg("bmi").toFloat();
    laptop_ip = server.arg("server");
    
    server.send(200, "text/plain", "Test Started. ESP32 is ready.");
    is_testing = true;
    drop_count = 0;
    Serial.println("Test triggered from laptop. Starting to monitor drops...");
  } else {
    server.send(400, "text/plain", "Missing parameters");
  }
}

// Function that handles the sensing and data upload
void performTest() {
  // ==========================================
  // SENSOR READING AND PINGING (/ping)
  // ==========================================
  // This is a simulation loop. You would replace this delay 
  // with actual accelerometer drop detection logic using `mpu.getEvent(&a, &g, &temp);`
  
  for(int i = 1; i <= max_drops; i++) {
    delay(2000); // Simulate the time it takes to detect a "heel drop"
    drop_count = i;
    Serial.printf("Drop %d detected\n", drop_count);
    
    // Ping the Flask server so the UI updates
    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      String ping_url = "http://" + laptop_ip + ":5000/ping?drop=" + String(drop_count);
      http.begin(ping_url);
      http.GET();
      http.end();
    }
  }

  // ==========================================
  // FINAL DATA UPLOAD (/upload)
  // ==========================================
  // Simulated final calculated parameters
  float peak_g = 2.5;
  float std_peak = 0.1;
  float decay = 0.8;
  float std_decay = 0.05;
  float freq_hz = 15.0;

  Serial.println("Test complete. Uploading data...");
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String upload_url = "http://" + laptop_ip + ":5000/upload";
    http.begin(upload_url);
    http.addHeader("Content-Type", "application/json");

    // Construct the JSON payload required by app.py
    StaticJsonDocument<200> doc;
    doc["id"] = patient_id;
    doc["age"] = age;
    doc["bmi"] = bmi;
    doc["peak"] = peak_g;
    doc["stdPeak"] = std_peak;
    doc["decay"] = decay;
    doc["stdDecay"] = std_decay;
    doc["freq"] = freq_hz;

    String jsonStr;
    serializeJson(doc, jsonStr);

    int httpResponseCode = http.POST(jsonStr);
    if (httpResponseCode > 0) {
      Serial.printf("Upload successful! Response code: %d\n", httpResponseCode);
    } else {
      Serial.printf("Error on upload: %s\n", http.errorToString(httpResponseCode).c_str());
    }
    http.end();
  }

  is_testing = false;
  Serial.println("ESP32 back to standby.");
}
