"""Flask app

Test application to experiment with Flask APIs. These APIs
will be used to serve inferences to predictive models
"""
import pickle

from flask import Flask, jsonify, request, make_response, abort
from dataclasses import dataclass
import dataclasses
from marshmallow import EXCLUDE, fields, pre_dump, Schema, validate, ValidationError

import desert
import pandas as pd

app = Flask(__name__)


class Model:
    """Object to store model while running"""

    model = None

    @classmethod
    def get_model(cls):
        with open("/workspaces/flask-docker-demo/model.obj", "rb") as f:
            cls.model = pickle.load(f)

    @classmethod
    def predict(cls, data):
        if cls.model == None:
            # cls.get_model()
            pass

        data["probability"] = 0.9
        data["label"] = "Accept"
        #  data["probability"] = cls.model.predict(data)
        return data


@dataclass
class ScorecardRequest:
    """Request dataclass to validate API inputs"""

    name: str = dataclasses.field(
        metadata=desert.metadata(
            fields.Str(
                required=True, validate=validate.Length(min=1, error="Name field cannot be empty")
            )
        )
    )
    id: str = dataclasses.field(
        metadata=desert.metadata(
            fields.Str(
                required=True,
                validate=validate.Length(min=0, error="ID field cannot be empty"),
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
        pass


@dataclass
class ScorecardResponse:
    """Response dataclass to validate API inputs"""

    id: str = dataclasses.field(
        metadata=desert.metadata(
            fields.Str(
                required=True,
                validate=validate.Length(min=1, error="ID field cannot be empty"),
            )
        )
    )
    probability: float = dataclasses.field(
        metadata=desert.metadata(
            fields.Float(
                required=True,
                validate=validate.Range(min=0, max=1, error="Probability field must be [0,1]"),
            )
        )
    )
    label: str = dataclasses.field(
        metadata=desert.metadata(
            fields.Str(
                required=True,
                validate=validate.OneOf(
                    choices=["Accept", "Fail"], error="label field must either be Accept or Fail"
                ),
            )
        )
    )

    def __post_init__(self):
        """
        Test
        """
        pass


@app.route("/ping", methods=["GET"])
def ping():
    """Function to check if API is working"""
    return make_response("pong!", 200)


@app.route("/predict", methods=["GET"])
def predict():
    validated_request = desert.schema(ScorecardRequest, meta={"unknown": EXCLUDE})
    validated_response = desert.schema(ScorecardResponse, meta={"unknown": EXCLUDE})

    request_data = request.get_json()

    try:
        policy = validated_request.load(request_data)
    except ValidationError as err:
        abort(jsonify(err.messages))

    data = pd.DataFrame(policy.__dict__, index=[0])

    data = Model.predict(data)

    try:
        scored_policy = validated_response.load(data.to_dict(orient="records")[0])
    except ValidationError as err:
        abort(jsonify(err.messages))

    return jsonify(scored_policy.__dict__)


if __name__ == "__main__":
    app.run(debug=True)
