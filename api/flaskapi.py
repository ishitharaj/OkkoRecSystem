import sys
sys.path.insert(1, '/app')
from models.lfm import LFMModel
from models.ranker import Ranker

from utils.utils import JsonEncoder
from models.pipeline import get_recommendations
from flask import Flask, request, jsonify, current_app, make_response
import json


# init application
app = Flask(__name__)

with app.app_context():
    lfm_model =  LFMModel()
    ranker = Ranker()

# set url to get predictions
@app.route('/get_recommendation')
def run():
    user_id = int(request.args.get('user_id'))
    top_k = int(request.args.get('top_k', 20))
    response = get_recommendations(
        user_id = user_id,
        lfm_model = lfm_model,
        ranker = ranker,
        top_k = top_k
    )
    
    response = jsonify(response)
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response
    # return json.dumps(response,ensure_ascii=False, cls = JsonEncoder)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)