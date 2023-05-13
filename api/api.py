import json
import logging
from flask import Flask, request
from inference import get_recommendations

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route('/predict')
def access_param():
    id = request.args.get('id')
    response = get_recommendations(int(id))
    converted_response = { str(k):float(v) for k, v in response.items()}
    return json.dumps(converted_response)

app.run(debug=True, host="0.0.0.0", port=5000)

# http://127.0.0.1:5000/predict?id=973171
# CHECK THAT IT WORKS!