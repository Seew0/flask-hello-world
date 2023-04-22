import numpy as np
import pickle

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
	return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
	float_features = [3000]
	features = [np.array(float_features)]
	prediction = model.predict(features)

	return dict(enumerate(prediction.flatten(), 1))


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000)
