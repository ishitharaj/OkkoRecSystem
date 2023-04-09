# TODO
# WRITE PIPELINE FOR DATA PREPARATION IN HERE TO USE FOR RANKER TRAININIG PIPELINE
from typing import Any, Dict, List
from configs.config import settings
import pandas as pd
import datetime as dt


def prepare_data_for_train(paths_config: Dict[str, str]):
    """
    function to prepare data to train catboost classifier.
    Basically, you have to wrap up code from full_recsys_pipeline.ipynb
    where we prepare data for classifier. In the end, it should work such
    that we trigger and use fit() method from ranker.py

        paths_config: dict, wher key is path name and value is the path to data
    """
    
    paths_config = {
        "interactions_data": "artefacts\data\interactions_df.csv",
        "users_data": "artefacts\data\items.csv",
        "items_data": "artefacts\data\items.csv",
    }
    
    users_data = pd.read_csv(paths_config["users_data"])
    movies_data = pd.read_csv(paths_config["items_data"])
    interactions_data = pd.read_csv(paths_config["interactions_data"])
    
    # # remove redundant data points
    # interactions_filtered = interactions_data.loc[interactions_data['total_dur'] > 300].reset_index(drop = True)
    
    # interactions_filtered['last_watch_dt'] = pd.to_datetime(interactions_filtered['last_watch_dt'])
    
    # # set dates params for filter
    # MAX_DATE = interactions_filtered['last_watch_dt'].max()
    # MIN_DATE = interactions_filtered['last_watch_dt'].min()
    # TEST_INTERVAL_DAYS = 14
    
    # TEST_MAX_DATE = MAX_DATE - dt.timedelta(days = TEST_INTERVAL_DAYS)
    
    # # define global train and test
    # global_train = interactions_filtered.loc[interactions_filtered['last_watch_dt'] < TEST_MAX_DATE]
    # global_test = interactions_filtered.loc[interactions_filtered['last_watch_dt'] >= TEST_MAX_DATE]
    
    # # now, we define "local" train and test to use some part of the global train for ranker
    # local_train_thresh = global_train['last_watch_dt'].quantile(q = .7, interpolation = 'nearest')
    
    # global_train = global_train.dropna().reset_index(drop = True)
    
    # local_train = global_train.loc[global_train['last_watch_dt'] < local_train_thresh]
    # local_test = global_train.loc[global_train['last_watch_dt'] >= local_train_thresh]
    
    # # finally, we will focus on warm start -- remove cold start users
    # local_test = local_test.loc[local_test['user_id'].isin(local_train['user_id'].unique())]
    
    # # joins user features
    # cbm_train_set = pd.merge(cbm_train_set, users_data[['user_id'] + settings.USER_FEATURES],
    #                         how = 'left', on = ['user_id'])
    # cbm_test_set = pd.merge(cbm_test_set, users_data[['user_id'] + settings.USER_FEATURES],
    #                         how = 'left', on = ['user_id'])
    
    
    return x_train, y_train, x_test, y_test
    
    


def get_items_features(item_ids: List[int], item_cols: List[str]) -> Dict[int, Any]:
    """
    function to get items features from our available data
    that we used in training (for all candidates)
        :item_ids:  item ids to filter by
        :item_cols: feature cols we need for inference

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
    pass


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