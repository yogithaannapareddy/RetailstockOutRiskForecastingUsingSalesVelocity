from flask import Flask, render_template, request
import pandas as pd
import os
from model import analyze_customer

# -----------------------------
# Flask App Configuration
# -----------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

dataframe = None

# -----------------------------
# Upload Dataset Page
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def upload():
    global dataframe

    if request.method == "POST":
        file = request.files["file"]

        if file.filename == "":
            return render_template("upload.html", error="No file selected")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Load dataset
        dataframe = pd.read_excel(file_path)

        return render_template("predict.html")

    return render_template("upload.html")

# -----------------------------
# Customer Prediction Page
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if dataframe is None:
        return render_template("result.html", error="Dataset not uploaded")

    customer_id = int(request.form["customer_id"])
    result = analyze_customer(dataframe, customer_id)

    if result is None:
        return render_template("result.html", error="Customer ID not found")

    return render_template("result.html", result=result,velocity_data=result["velocity_trend"])

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)