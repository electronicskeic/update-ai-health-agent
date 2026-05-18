# AI Health Agent (BMI + Diabetes Risk & Recommendations)

This project is an AI-powered health monitoring and recommendation application. It trains machine learning models locally on specialized datasets to provide actionable insights into obesity and diabetes risks, and leverages Google Gemini for personalized health coaching!

## Features

- **Obesity Risk Analyzer**: Calculates your BMI and predicts obesity risk using a Logistic Regression model trained on `bmi.xlsx`. 
- **Diabetes Risk Analyzer**: Predicts the user's probability of developing diabetes using the renowned Pima Indians dataset (`diabetes.csv`).
- **Interactive Insights**: Visualizes correlation charts, BMI distributions, and dataset visualizations using dynamic Plotly charts.
- **Personalized Planning**: Generates targeted diet and fitness recommendations based on your unique risk profiles.
- **Account System & Check-ins**: Built-in persistent SQLite database for profiles, tracking weight progress over time, and saving your chat history.
- **AI Health Coach (Gemini)**: Features an integrated chatbot powered by Google's Gemini models. Gemini evaluates your specific BMI, diabetes risk, and local statistics to give customized health advice.

## Setup Instructions

1. Ensure you have Python installed.
2. Create and activate a Virtual Environment (recommended).
3. Install the dependencies:
```bash
python -m pip install -r requirements.txt
```

## Adding your Gemini API Key

The chatbot relies on Google's Gemini capabilities. You can set up your key safely using Streamlit Secrets.

1. Ensure you have a `.streamlit` folder at the root of the project.
2. Inside `.streamlit`, create a file called `secrets.toml`.
3. Add your key exactly like this:
```toml
GEMINI_API_KEY = "sk-your-gemini-key-goes-here"
```

*(Alternatively, you can just set `GEMINI_API_KEY` in your terminal as an environment variable before running).*

## Running the Application

Once your dependencies are installed, you can launch the app locally:

```bash
streamlit run app.py
```

## Datasets

The application relies on two dataset files to train its local models at runtime:
- `bmi.xlsx`: Used to map height, weight, and age against obesity incidence.
- `diabetes.csv`: The Pima Indians Diabetes Database, used to compute diabetes predisposition based on clinical metrics (glucose, insulin, skin thickness, etc.).
