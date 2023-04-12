import pandas as pd
import numpy as np
import datetime as dt
from typing import Any, Dict, List


def read_csv_from_gdrive(url):
    """
    gets csv data from a given url (from file -> share -> copy link)
    :url: *****/view?usp=share_link
    """
    file_id = url.split("/")[-2]
    file_path = "https://drive.google.com/uc?export=download&id=" + file_id
    data = pd.read_csv(file_path)

    return data

def prepare_interaction_data(paths_config: Dict[str, str]):
    users_data = pd.read_csv(paths_config["users_data"])
    movies_metadata = pd.read_csv(paths_config["items_data"])
    interactions = pd.read_csv(paths_config["interactions_data"])
    
    # crate mapper for movieId and title names
    item_name_mapper = dict(zip(movies_metadata['item_id'], movies_metadata['title']))
    
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
    
    print(global_train.shape, global_test.shape)
    
    local_train_thresh = global_train['last_watch_dt'].quantile(q = .7, interpolation = 'nearest')
    global_train = global_train.dropna().reset_index(drop = True)
    
    local_train = global_train.loc[global_train['last_watch_dt'] < local_train_thresh]
    local_test = global_train.loc[global_train['last_watch_dt'] >= local_train_thresh]
    
    # finally, we will focus on warm start -- remove cold start users
    local_test = local_test.loc[local_test['user_id'].isin(local_train['user_id'].unique())]
    
    return global_train, global_test, local_train, local_test
    

def generate_lightfm_recs_mapper(
        model: object,
        item_ids: list,
        known_items: dict,
        user_features: list,
        item_features: list,
        N: int,
        user_mapping: dict,
        item_inv_mapping: dict,
        num_threads: int = 4
        ):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.predict(
            user_id,
            item_ids,
            user_features = user_features,
            item_features = item_features,
            num_threads = num_threads)
        
        additional_N = len(known_items[user_id]) if user_id in known_items else 0
        total_N = N + additional_N
        top_cols = np.argpartition(recs, -np.arange(total_N))[-total_N:][::-1]
        
        final_recs = [item_inv_mapping[item] for item in top_cols]
        if additional_N > 0:
            filter_items = known_items[user_id]
            final_recs = [item for item in final_recs if item not in filter_items]
        return final_recs[:N]
    return _recs_mapper