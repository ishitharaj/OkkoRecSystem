import logging
import pandas as pd
import pickle

# import sys
# sys.path.append(r"C:\Users\Ishitha\Desktop\dev\OkkoRecSystem")

logging.basicConfig(level=logging.INFO)

from configs.config import settings
from data_prep.prepare_ranker_data import (
    get_items_features,
    get_user_features,
    prepare_ranker_input,
)

from models.lfm import LFMModel
from models.ranker import Ranker

lfm_model =  LFMModel()
ranker = Ranker()

def get_recommendations(
    user_id: int, lfm_model: object, ranker: object, top_k: int = 20
):
    """
    function to get recommendation for a given user id
    """

    try:
        logging.info("getting 1st level candidates")
        candidates = lfm_model.infer(user_id=user_id, top_k=top_k)

        logging.info("getting features...")
        user_features = get_user_features(user_id, user_cols=settings.USER_FEATURES)
        item_features = get_items_features(
            item_ids=list(candidates.keys()), item_cols=settings.ITEM_FEATURES
        )
        print(user_features)
        print(item_features)

        ranker_input = prepare_ranker_input(
            candidates=candidates,
            item_features=item_features,
            user_features=user_features,
            ranker_features_order=ranker.ranker.feature_names_,
        )
        preds = ranker.infer(ranker_input=ranker_input)
        predictions = dict(zip(candidates.keys(), preds))
        sorted_recommendations = dict(
            sorted(predictions.items(), key=lambda item: item[1], reverse=True)
        )
        
        recs_df = pd.DataFrame(columns=['movie_id', 'title'])
        recs_df['movie_id'] = [key for key in sorted_recommendations.keys()]
        with open('artefacts/item_name_mapper_data.pkl', 'rb') as fp:
            items_data = pickle.load(fp)
            recs_df['title'] = recs_df['movie_id'].map(items_data)

        output = {
            "recommendations": list(sorted_recommendations.keys()),
            "recommendations_titles": list(recs_df['title']),
            "status": "success",
            "msg": None,
        }

    except Exception as e:
        output = {"recommendations": None, "recommendations_titles": None, "status": "error", "msg": str(e)}

    return output

# if __name__ == '__main__':
#     response = get_recommendations(
#         user_id = 228019116,
#         lfm_model = lfm_model,
#         ranker = ranker
#     )
#     print(response)