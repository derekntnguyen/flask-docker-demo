"""Flask app

Test application to experiment with Flask APIs. These APIs
will be used to serve inferences to predictive models
"""

from flask import Flask, jsonify, request, make_response, abort
from dataclasses import dataclass
import dataclasses
from marshmallow import EXCLUDE, fields, pre_dump, Schema, validate, ValidationError

import desert
import pandas as pd

app = Flask(__name__)


@dataclass
class Scorecard:
    name: str
    age: int
    is_alive: bool


class ScorecardSchema(Schema):
    name = fields.Str(required=True)
    age = fields.Int(
        required=True,
        validate=validate.Range(
            min=0, 
            error="Cannot be a negative age",
            ),
        )
    is_alive = fields.Bool(required=True)


@app.route("/ping", methods=["GET"])
def ping():
    """Function to check if API is working"""
    return make_response("pong!", 200)


@app.route("/predict", methods=["GET"])
def predict():
    data = request.get_json()
    
    try:
        validated_response = ScorecardSchema(unknown=EXCLUDE).load(data)
    except ValidationError as err:
        abort(jsonify(err.messages))


    policy = Scorecard(**validated_response)

    return jsonify(policy.__dict__)

    # if not data:
    #     abort("Invalid data")

    # df = pd.DataFrame(data["data"])
    # df["score"] = 0.8
    # predictions = df[["id", "score"]]
    # response = predictions.to_json(orient="split")

    # return make_response(response, 200)


if __name__ == "__main__":
    app.run(debug=True)
