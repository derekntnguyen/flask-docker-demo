"""Flask app

Test application to experiment with Flask APIs. These APIs
will be used to serve inferences to predictive models
"""

from flask import Flask, jsonify, request, make_response
import pandas as pd

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Function to check if API is working"""
    return make_response("pong!", 200)

@app.route("/predict", methods=["GET"])
def predict():
    data = request.get_json()

    if not data:
        return make_response("Invalid data", 404)

    df = pd.DataFrame(data["data"])
    df["score"] = 0.8
    predictions = df[["id", "score"]]
    response = predictions.to_json(orient="split")

    return make_response(response, 200)


if __name__ == "__main__":
    app.run(debug=True)