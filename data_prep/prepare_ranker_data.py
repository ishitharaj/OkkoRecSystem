# # TODO
# # WRITE PIPELINE FOR DATA PREPARATION IN HERE TO USE FOR RANKER TRAININIG PIPELINE
# from json import loads, dumps
# import pandas as pd

# from typing import Any, Dict, List
# from configs.config import settings
# import pandas as pd
# import datetime as dt
# from sklearn.utils import shuffle
# from lightfm.data import Dataset
# from sklearn.model_selection import train_test_split
# from utils.utils import generate_lightfm_recs_mapper, prepare_interaction_data


# def prepare_data_for_train(paths_config: Dict[str, str], lfm_model, global_train, global_test, local_train, local_test):
#     """
#     function to prepare data to train catboost classifier.
#     Basically, you have to wrap up code from full_recsys_pipeline.ipynb
#     where we prepare data for classifier. In the end, it should work such
#     that we trigger and use fit() method from ranker.py
#         paths_config: dict, wher key is path name and value is the path to data
#     """
#     users_data = pd.read_csv(paths_config["users_data"])
#     movies_metadata = pd.read_csv(paths_config["items_data"])
    
#     # crate mapper for movieId and title names
#     item_name_mapper = dict(zip(movies_metadata['item_id'], movies_metadata['title']))
    
#     # init class
#     dataset = Dataset()

#     # fit tuple of user and movie interactions
#     dataset.fit(local_train['user_id'].unique(), local_train['item_id'].unique())
    
#     lightfm_mapping = dataset.mapping()
    
#     lightfm_mapping = {
#         'users_mapping': lightfm_mapping[0],
#         'user_features_mapping': lightfm_mapping[1],
#         'items_mapping': lightfm_mapping[2],
#         'item_features_mapping': lightfm_mapping[3],
#     }
    
#     lightfm_mapping['users_inv_mapping'] = {v: k for k, v in lightfm_mapping['users_mapping'].items()}
#     lightfm_mapping['items_inv_mapping'] = {v: k for k, v in lightfm_mapping['items_mapping'].items()}
    
#     top_N = 10
#     all_cols = list(lightfm_mapping['items_mapping'].values())
    
#     # crate mapper for movieId and title names
#     item_name_mapper = dict(zip(movies_metadata['item_id'], movies_metadata['title']))
    
#     # let's make predictions for all users in test
#     test_preds = pd.DataFrame({
#         'user_id': local_test['user_id'].unique()
#     })
    
#     mapper = generate_lightfm_recs_mapper(
#         lfm_model, 
#         item_ids = all_cols, 
#         known_items = dict(),
#         N = top_N,
#         user_features = None, 
#         item_features = None, 
#         user_mapping = lightfm_mapping['users_mapping'],
#         item_inv_mapping = lightfm_mapping['items_inv_mapping'],
#         num_threads = 20
#         )
    
#     # let's make predictions for all users in test
#     test_preds = pd.DataFrame({
#         'user_id': local_test['user_id'].unique()
#     })
    
    
#     test_preds['item_id'] = test_preds['user_id'].map(mapper)
#     test_preds = test_preds.explode('item_id')
#     test_preds['rank'] = test_preds.groupby('user_id').cumcount() + 1 
#     test_preds['item_name'] = test_preds['item_id'].map(item_name_mapper)
    
    
#     ''' create 0/1 as indication of interaction:
#          > positive event -- 1, if watch_pct is not null
#          > negative event -- 0 otherwise
#     '''
#     positive_preds = pd.merge(test_preds, local_test, how = 'inner', on = ['user_id', 'item_id'])
#     positive_preds['target'] = 1
    
#     negative_preds = pd.merge(test_preds, local_test, how = 'left', on = ['user_id', 'item_id'])
#     negative_preds = negative_preds.loc[negative_preds['watched_pct'].isnull()].sample(frac = .2)
#     negative_preds['target'] = 0
    
#     # random split to train ranker
#     train_users, test_users = train_test_split(
#         local_test['user_id'].unique(),
#         test_size = .2,
#         random_state = 13
#         )
    
#     cbm_train_set = shuffle(
#         pd.concat(
#         [positive_preds.loc[positive_preds['user_id'].isin(train_users)],
#         negative_preds.loc[negative_preds['user_id'].isin(train_users)]]
#         )
#     )
    
#     cbm_test_set = shuffle(
#         pd.concat(
#         [positive_preds.loc[positive_preds['user_id'].isin(test_users)],
#         negative_preds.loc[negative_preds['user_id'].isin(test_users)]]
#         )
#     )
    
#     # joins user features
#     cbm_train_set = pd.merge(cbm_train_set, users_data[['user_id'] + settings.USER_FEATURES],
#                             how = 'left', on = ['user_id'])
#     cbm_test_set = pd.merge(cbm_test_set, users_data[['user_id'] + settings.USER_FEATURES],
#                             how = 'left', on = ['user_id'])
    
#     # joins item features
#     cbm_train_set = pd.merge(cbm_train_set, movies_metadata[['item_id'] + settings.ITEM_FEATURES],
#                             how = 'left', on = ['item_id'])
#     cbm_test_set = pd.merge(cbm_test_set, movies_metadata[['item_id'] + settings.ITEM_FEATURES],
#                             how = 'left', on = ['item_id'])
    
#     x_train, y_train = cbm_train_set.drop(settings.ID_COLS + settings.DROP_COLS + settings.TARGET, axis = 1), cbm_train_set[settings.TARGET]
#     x_test, y_test = cbm_test_set.drop(settings.ID_COLS + settings.DROP_COLS + settings.TARGET, axis = 1), cbm_test_set[settings.TARGET]
    
#     x_train = x_train.fillna(x_train.mode().iloc[0])
#     x_test = x_test.fillna(x_test.mode().iloc[0])
    
#     return x_train, y_train, x_test, y_test
    
    


# def get_items_features(item_ids: List[int], item_cols: List[str]) -> Dict[int, Any]:
#     """
#     function to get items features from our available data
#     that we used in training (for all candidates)
#         :item_ids:  item ids to filter by
#         :item_cols: feature cols we need for inference
#     """

#     paths_config = {
#         "interactions_data": "artefacts/data/interactions_df.csv",
#         "users_data": "artefacts/data/users.csv",
#         "items_data": "artefacts/data/items.csv"
#         }

#     movies_data = pd.read_csv(paths_config["items_data"])
#     item_df = movies_data[movies_data['item_id'].isin(item_ids)][['item_id',
#                                                                   'content_type',
#                                                                   'release_year',
#                                                                   'for_kids',
#                                                                   'age_rating']]
#     item_df.set_index('item_id', inplace=True)
#     output = item_df.to_dict('index')

#     """
#     EXAMPLE OUTPUT
#     {
#     9169: {
#     'content_type': 'film',
#     'release_year': 2020,
#     'for_kids': None,
#     'age_rating': 16
#         },

#     10440: {
#     'content_type': 'series',
#     'release_year': 2021,
#     'for_kids': None,
#     'age_rating': 18
#         }
#     }

#     """
#     return output


# def get_user_features(user_id: int, user_cols: List[str]) -> Dict[str, Any]:
#     """
#     function to get user features from our available data
#     that we used in training
#         :user_id: user id to filter by
#         :user_cols: feature cols we need for inference
#     EXAMPLE OUTPUT
#     {
#         'age': None,
#         'income': None,
#         'sex': None,
#         'kids_flg': None
#     }
#     """
    
#     paths_config = {
#         "interactions_data": "artefacts/data/interactions_df.csv",
#         "users_data": "artefacts/data/users.csv",
#         "items_data": "artefacts/data/items.csv"
#         }
#     users_data = pd.read_csv(paths_config["users_data"])
#     return users_data.loc[users_data['user_id'] == user_id][user_cols].to_dict('records')[0]


# def prepare_ranker_input(
#     candidates: Dict[int, int],
#     item_features: Dict[int, Any],
#     user_features: Dict[int, Any],
#     ranker_features_order,
# ):
#     ranker_input = []
#     for k in item_features.keys():
#         item_features[k].update(user_features)
#         item_features[k]["rank"] = candidates[k]
#         item_features[k] = {
#             feature: item_features[k][feature] for feature in ranker_features_order
#         }
#         ranker_input.append(list(item_features[k].values()))

#     return ranker_input
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from configs.config import settings
from utils.utils import (
    generate_lightfm_recs_mapper,
    load_model,
    read_parquet_from_gdrive,
)


def prepare_data_for_train() -> Tuple[pd.DataFrame]:
    """
    function to prepare data to train catboost classifier.
    Basically, you have to wrap up code from full_recsys_pipeline.ipynb
    where we prepare data for classifier. In the end, it should work such
    that we trigger and use fit() method from ranker.py
        paths_config: dict, wher key is path name and value is the path to data
    """
    # load model artefacts
    model = load_model(settings.LFM_TRAIN_PARAMS.MODEL_PATH)
    dataset = load_model(settings.LFM_TRAIN_PARAMS.MAPPER_PATH)

    # get users sample to predict candidates on
    user_sample_path = settings.RANKER_DATA.USERS_SAMPLE
    if "https" in user_sample_path:
        users_sample = read_parquet_from_gdrive(user_sample_path)
    else:
        users_sample = pd.read_parquet(user_sample_path)

    # pred candidates
    preds = users_sample[["user_id"]].drop_duplicates().reset_index(drop=True)

    # init mapper with model
    item_ids = list(dataset.mapping()[2].values())
    item_inv_mapper = {v: k for k, v in dataset.mapping()[2].items()}
    mapper = generate_lightfm_recs_mapper(
        model,
        item_ids=item_ids,
        known_items=dict(),
        N=settings.LFM_PREDS_PARAMS.TOP_K,
        user_features=None,
        item_features=None,
        user_mapping=dataset.mapping()[0],
        item_inv_mapping=item_inv_mapper,
        num_threads=20,
    )
    preds["item_id"] = preds["user_id"].map(mapper)

    # define target & prepare ranker sample
    preds = preds.explode("item_id")
    logging.info(f"Shape of the preds: {preds.shape}")

    preds["rank"] = preds.groupby("user_id").cumcount() + 1
    logging.info(
        f"Number of unique candidates generated by the LFM: {preds.item_id.nunique()}"
    )

    train_data = get_ranker_sample(preds=preds, users_sample=users_sample)

    return train_data


def get_ranker_sample(preds: pd.DataFrame, users_sample: pd.DataFrame):
    """
    final step to use candidates generation and users interaction to define
    train data - join features, define target, split into train & test samples
    """
    local_test = users_sample.copy(deep=True)

    # prepare train & test
    positive_preds = pd.merge(preds, local_test, how="inner", on=["user_id", "item_id"])
    positive_preds["target"] = 1
    logging.info(f"Shape of the positive target preds: {positive_preds.shape}")

    negative_preds = pd.merge(preds, local_test, how="left", on=["user_id", "item_id"])
    negative_preds = negative_preds.loc[negative_preds["watched_pct"].isnull()].sample(
        frac=settings.RANKER_DATA.NEG_FRAC
    )
    negative_preds["target"] = 0
    logging.info(f"Shape of the negative target preds: {positive_preds.shape}")

    # random split to train ranker
    train_users, test_users = train_test_split(
        local_test["user_id"].unique(),
        test_size=0.2,
        random_state=settings.RANKER_DATA.RANDOM_STATE,
    )

    cbm_train_set = shuffle(
        pd.concat(
            [
                positive_preds.loc[positive_preds["user_id"].isin(train_users)],
                negative_preds.loc[negative_preds["user_id"].isin(train_users)],
            ]
        )
    )

    cbm_test_set = shuffle(
        pd.concat(
            [
                positive_preds.loc[positive_preds["user_id"].isin(test_users)],
                negative_preds.loc[negative_preds["user_id"].isin(test_users)],
            ]
        )
    )

    users_data = read_parquet_from_gdrive(settings.RANKER_DATA.USERS_DATA_PATH)
    items_data = read_parquet_from_gdrive(settings.RANKER_DATA.MOVIES_DATA_PATH)

    # join user features
    cbm_train_set = pd.merge(
        cbm_train_set,
        users_data[["user_id"] + settings.USER_FEATURES],
        how="left",
        on=["user_id"],
    )
    cbm_test_set = pd.merge(
        cbm_test_set,
        users_data[["user_id"] + settings.USER_FEATURES],
        how="left",
        on=["user_id"],
    )
    # join item features
    cbm_train_set = pd.merge(
        cbm_train_set,
        items_data[["item_id"] + settings.ITEM_FEATURES],
        how="left",
        on=["item_id"],
    )
    cbm_test_set = pd.merge(
        cbm_test_set,
        items_data[["item_id"] + settings.ITEM_FEATURES],
        how="left",
        on=["item_id"],
    )

    # final steps
    drop_cols = (
        settings.CBM_COLS_CONFIG.ID_COLS.to_list()
        + settings.CBM_COLS_CONFIG.DROP_COLS.to_list()
        + settings.TARGET.to_list()
    )
    X_train, y_train = (
        cbm_train_set.drop(
            drop_cols,
            axis=1,
        ),
        cbm_train_set[settings.TARGET],
    )
    X_test, y_test = (
        cbm_test_set.drop(
            drop_cols,
            axis=1,
        ),
        cbm_test_set[settings.TARGET],
    )
    logging.info(X_train.shape, X_test.shape)

    # no time dependent feature -- we can leave it with mode
    X_train = X_train.fillna(X_train.mode().iloc[0])
    X_test = X_test.fillna(X_test.mode().iloc[0])

    return X_train, X_test, y_train, y_test


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
    item_features = read_parquet_from_gdrive(
        "https://drive.google.com/file/d/1XGLUhHpwr0NxU7T4vYNRyaqwSK5HU3N4/view?usp=share_link"
    )
    item_features = item_features.set_index("item_id")
    item_features = item_features.to_dict("index")

    # collect all items
    output = {}
    for id in item_ids:
        output[id] = {k: v for k, v in item_features.get(id).items() if k in item_cols}

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
    users = read_parquet_from_gdrive(
        "https://drive.google.com/file/d/1MCTl6hlhFYer1BTwjzIBfdBZdDS_mK8e/view?usp=share_link"
    )
    users = users.set_index("user_id")
    users_dict = users.to_dict("index")
    return {k: v for k, v in users_dict.get(user_id).items() if k in user_cols}


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