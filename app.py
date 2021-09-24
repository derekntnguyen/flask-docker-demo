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
    name: str = dataclasses.field(
        metadata=desert.metadata(
            fields.Str(
                required=True, validate=validate.Length(min=1, error="Name field cannot be empty")
            )
        )
    )
    age: int = dataclasses.field(
        metadata=desert.metadata(
            fields.Int(
                required=True,
                validate=validate.Range(min=0, error="Age field cannot be negative"),
            )
        )
    )
    is_alive: bool = dataclasses.field(
        metadata=desert.metadata(
            fields.Bool(
                required=True,
            )
        )
    )

    def __post_init__(self):
        """
        Test
        """
        self.age = self.age + 1


@app.route("/ping", methods=["GET"])
def ping():
    """Function to check if API is working"""
    return make_response("pong!", 200)


@app.route("/predict", methods=["GET"])
def predict():
    data = request.get_json()
    validated_response = desert.schema(Scorecard, meta={"unknown": EXCLUDE})

    try:
        policy = validated_response.load(data)
    except ValidationError as err:
        abort(jsonify(err.messages))

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
