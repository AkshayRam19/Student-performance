from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import logging

app = Flask(__name__)


logging.basicConfig(level=logging.DEBUG)

model = pickle.load(open("best_student_model.sav", 'rb'))
encoder = pickle.load(open("student_encoder.sav", 'rb'))
scaler = pickle.load(open("student_scaler.sav", 'rb'))

input_columns = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
    "ML Mid term exam",
    "ES Mid term exam",
    "CC Mid term exam",
    "CN Mid term exam",
    "BDA Mid term exam"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        app.logger.debug(f"Form data received: {data}")
        df = pd.DataFrame([data])
        
        for col in df.columns:
            if col in ["Extracurricular Activities"]:
                df[col] = encoder.transform(df[col])
            else:
                df[col] = df[col].astype(float)

        df_scaled = scaler.transform(df)

        
        prediction = model.predict(df_scaled)[0]

        app.logger.debug(f"Prediction: {prediction}")

        return render_template('predict.html', prediction=round(prediction, 2))
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return str(e), 500

@app.route('/graphs')
def graphs():
    try:
        data = pd.read_csv("Student_Performance_.csv")
        app.logger.debug("CSV file loaded successfully.")

        graph_data = {}
        for column in input_columns:
            plt.figure(figsize=(8, 6))
            plt.scatter(data[column], data["Performance Index"])
            plt.xlabel(column)
            plt.ylabel("Performance Index")
            plt.title(f'Performance Index vs {column}')
            plt.grid(True)
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = 'data:image/png;base64,' + string.decode('utf-8')
            graph_data[column] = uri
            plt.close()

        return render_template('graph.html', graph_data=graph_data)
    except Exception as e:
        app.logger.error(f"Error generating graphs: {e}")
        return str(e), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
