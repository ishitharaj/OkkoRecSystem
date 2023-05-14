import sys
sys.path.insert(1, '/app')
from models.lfm import LFMModel
from models.ranker import Ranker

from utils.utils import JsonEncoder
from models.pipeline import get_recommendations
from flask import Flask, request
import json

# init application
app = Flask(__name__)

with app.app_context():
    lfm_model =  LFMModel()
    ranker = Ranker()

# set url to get predictions
@app.route('/predict')
def run():
    user_id = int(request.args.get('user_id'))
    top_k = int(request.args.get('top_k', 20))
    response = get_recommendations(
        user_id = user_id,
        lfm_model = lfm_model,
        ranker = ranker,
        top_k = top_k
    )
    return json.dumps(response,ensure_ascii=False, cls = JsonEncoder)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
    
### http://127.0.0.1:8080/predict?user_id=973171
# CHECK THAT IT WORKS!

    
## --- Option 2
# import json
# import logging
# import pandas as pd
# import pickle
# from flask import Flask, request
# from inference import get_recommendations

# logging.basicConfig(level=logging.INFO)

# app = Flask(__name__)

# @app.route('/predict')
# def access_param():
#     id = request.args.get('id')
#     response = get_recommendations(int(id))
#     converted_response = {k:float(v) for k, v in response.items()}
#     recs_df = pd.DataFrame(columns=['movie_id', 'title'])
#     recs_df['movie_id'] = [key for key in converted_response.keys()]
#     with open('artefacts/item_name_mapper_data.pkl', 'rb') as fp:
#         items_data = pickle.load(fp)
#         recs_df['title'] = recs_df['movie_id'].map(items_data)
#     return json.dumps(list(recs_df['title']))

# app.run(debug=True, host="0.0.0.0", port=5000)

# http://127.0.0.1:5000/predict?id=973171
# CHECK THAT IT WORKS!