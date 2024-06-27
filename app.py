from flask import Flask, render_template, request
import urllib.request
import json
import os
import ssl

app = Flask(__name__)

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    sepal_length = float(request.form['sepal-length'])
    sepal_width = float(request.form['sepal-width'])
    petal_length = float(request.form['petal-length'])
    petal_width = float(request.form['petal-width'])

    # Prepare the data for the REST API request
    data = {
        "Inputs": {
            "data": [
                {
                    "SepalLengthCm": sepal_length,
                    "SepalWidthCm": sepal_width,
                    "PetalLengthCm": petal_length,
                    "PetalWidthCm": petal_width
                }
            ]
        },
        "GlobalParameters": {
            "method": "predict"
        }
    }
    
    body = str.encode(json.dumps(data))

    url = 'https://iris-gcjnq.centralindia.inference.ml.azure.com/score'
    api_key = 'hJ8snljIBkSlfMIwLpQWIYWpDOP2NEWZ'
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'class2-1' }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
       
        response_data = json.loads(result)
        if 'Results' in response_data:
            prediction = response_data['Results'][0]
            return f'Predicted Iris Class: {prediction}'
        else:
            return 'Unexpected response format from the Azure ML endpoint.'
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
        return 'An error occurred while making the prediction. Please check the console for more details.'

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")