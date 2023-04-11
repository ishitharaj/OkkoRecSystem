# TODO
# WRITE PIPELINE FOR DATA PREPARATION IN HERE TO USE FOR RANKER TRAININIG PIPELINE
from json import loads, dumps
import pandas as pd
from typing import Any, Dict, List
from configs.config import settings
import pandas as pd
import datetime as dt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def prepare_data_for_train(paths_config: Dict[str, str]):
    """
    function to prepare data to train catboost classifier.
    Basically, you have to wrap up code from full_recsys_pipeline.ipynb
    where we prepare data for classifier. In the end, it should work such
    that we trigger and use fit() method from ranker.py
        paths_config: dict, wher key is path name and value is the path to data
    """
    users_data = pd.read_csv(paths_config["users_data"])
    movies_metadata = pd.read_csv(paths_config["items_data"])
    interactions = pd.read_csv(paths_config["interactions_data"])
    
    # remove redundant data points
    interactions_filtered = interactions.loc[interactions['total_dur'] > 300].reset_index(drop = True)
    
    interactions_filtered['last_watch_dt'] = pd.to_datetime(interactions_filtered['last_watch_dt'])
    
    # set dates params for filter
    MAX_DATE = interactions_filtered['last_watch_dt'].max()
    TEST_INTERVAL_DAYS = 14
    
    TEST_MAX_DATE = MAX_DATE - dt.timedelta(days = TEST_INTERVAL_DAYS)
    
    # define global train and test
    global_train = interactions_filtered.loc[interactions_filtered['last_watch_dt'] < TEST_MAX_DATE]
    global_test = interactions_filtered.loc[interactions_filtered['last_watch_dt'] >= TEST_MAX_DATE]
    
    # joins user features
    cbm_train_set = pd.merge(global_train, users_data[['user_id'] + settings.USER_FEATURES],
                            how = 'left', on = ['user_id'])
    cbm_test_set = pd.merge(global_test, users_data[['user_id'] + settings.USER_FEATURES],
                            how = 'left', on = ['user_id'])
    
    # joins item features
    cbm_train_set = pd.merge(cbm_train_set, movies_metadata[['item_id'] + settings.ITEM_FEATURES],
                            how = 'left', on = ['item_id'])
    cbm_test_set = pd.merge(cbm_test_set, movies_metadata[['item_id'] + settings.ITEM_FEATURES],
                            how = 'left', on = ['item_id'])
        
    x_train, y_train = cbm_train_set.drop(settings.ID_COLS + settings.DROP_COLS + settings.TARGET, axis = 1), cbm_train_set[settings.TARGET]
    x_test, y_test = cbm_test_set.drop(settings.ID_COLS + settings.DROP_COLS + settings.TARGET, axis = 1), cbm_test_set[settings.TARGET]
    
    return x_train, y_train, x_test, y_test
    
    


def get_items_features(item_ids: List[int], item_cols: List[str]) -> Dict[int, Any]:
    """
    function to get items features from our available data
    that we used in training (for all candidates)
        :item_ids:  item ids to filter by
        :item_cols: feature cols we need for inference
    """

    paths_config = {
        "interactions_data": "artefacts\data\interactions_df.csv",
        "users_data": "artefacts\data\items.csv",
        "items_data": "artefacts\data\items.csv",
    }

    users_data = pd.read_csv(paths_config["users_data"])
    movies_data = pd.read_csv(paths_config["items_data"])
    interactions_data = pd.read_csv(paths_config["interactions_data"])


    item_df = movies_data[item_ids+item_cols]
    item_df.set_index(item_ids, inplace=True)
    result = item_df.to_json(orient="index")
    parsed = loads(result)
    output = dumps(parsed, indent=3)

    """
    EXAMPLE OUTPUT
    {
    9169: {
    'content_type': 'film',
    'release_year': 2020,
    'for_kids': None,
    'age_rating': 16
        },

    10440: {
    'content_type': 'series',
    'release_year': 2021,
    'for_kids': None,
    'age_rating': 18
        }
    }

    """
    return output


def get_user_features(user_id: int, user_cols: List[str]) -> Dict[str, Any]:
    """
    function to get user features from our available data
    that we used in training
        :user_id: user id to filter by
        :user_cols: feature cols we need for inference

    EXAMPLE OUTPUT
    {
        'age': None,
        'income': None,
        'sex': None,
        'kids_flg': None
    }
    """
    pass


def prepare_ranker_input(
    candidates: Dict[int, int],
    item_features: Dict[int, Any],
    user_features: Dict[int, Any],
    ranker_features_order,
):
    ranker_input = []
    for k in item_features.keys():
        item_features[k].update(user_features)
        item_features[k]["rank"] = candidates[k]
        item_features[k] = {
            feature: item_features[k][feature] for feature in ranker_features_order
        }
        ranker_input.append(list(item_features[k].values()))

    return ranker_input