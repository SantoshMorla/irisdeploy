import json
import pickle
import os
import numpy as np
from azureml.core.model import Model

def init():
    global rfmodel

    model_path = Model.get_model_path('random_forest_model')
    rfmodel = pickle.load(open(model_path, 'rb'))


def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        prediction = rfmodel.predict(data)
        return json.dumps({"prediction": prediction.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})