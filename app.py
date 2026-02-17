from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])

    features = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(features)

    price = round(prediction[0], 2)

    return render_template("index.html", prediction_text=f"Predicted Price: ₹ {price}")

if __name__ == "__main__":
    app.run(debug=True)
