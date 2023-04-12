# TODO
# WRITE PIPELINE FOR DATA PREPARATION IN HERE TO USE FOR RANKER TRAININIG PIPELINE
from json import loads, dumps
import pandas as pd

from typing import Any, Dict, List
from configs.config import settings
import pandas as pd
import datetime as dt
from sklearn.utils import shuffle
from lightfm.data import Dataset
from sklearn.model_selection import train_test_split
from utils.utils import generate_lightfm_recs_mapper, prepare_interaction_data


def prepare_data_for_train(paths_config: Dict[str, str], lfm_model, global_train, global_test, local_train, local_test):
    """
    function to prepare data to train catboost classifier.
    Basically, you have to wrap up code from full_recsys_pipeline.ipynb
    where we prepare data for classifier. In the end, it should work such
    that we trigger and use fit() method from ranker.py
        paths_config: dict, wher key is path name and value is the path to data
    """
    users_data = pd.read_csv(paths_config["users_data"])
    movies_metadata = pd.read_csv(paths_config["items_data"])
    
    # crate mapper for movieId and title names
    item_name_mapper = dict(zip(movies_metadata['item_id'], movies_metadata['title']))
    
    # init class
    dataset = Dataset()

    # fit tuple of user and movie interactions
    dataset.fit(local_train['user_id'].unique(), local_train['item_id'].unique())
    
    lightfm_mapping = dataset.mapping()
    
    lightfm_mapping = {
        'users_mapping': lightfm_mapping[0],
        'user_features_mapping': lightfm_mapping[1],
        'items_mapping': lightfm_mapping[2],
        'item_features_mapping': lightfm_mapping[3],
    }
    
    lightfm_mapping['users_inv_mapping'] = {v: k for k, v in lightfm_mapping['users_mapping'].items()}
    lightfm_mapping['items_inv_mapping'] = {v: k for k, v in lightfm_mapping['items_mapping'].items()}
    
    top_N = 10
    all_cols = list(lightfm_mapping['items_mapping'].values())
    
    # crate mapper for movieId and title names
    item_name_mapper = dict(zip(movies_metadata['item_id'], movies_metadata['title']))
    
    # let's make predictions for all users in test
    test_preds = pd.DataFrame({
        'user_id': local_test['user_id'].unique()
    })
    
    mapper = generate_lightfm_recs_mapper(
        lfm_model, 
        item_ids = all_cols, 
        known_items = dict(),
        N = top_N,
        user_features = None, 
        item_features = None, 
        user_mapping = lightfm_mapping['users_mapping'],
        item_inv_mapping = lightfm_mapping['items_inv_mapping'],
        num_threads = 20
        )
    
    # let's make predictions for all users in test
    test_preds = pd.DataFrame({
        'user_id': local_test['user_id'].unique()
    })
    
    
    test_preds['item_id'] = test_preds['user_id'].map(mapper)
    test_preds = test_preds.explode('item_id')
    test_preds['rank'] = test_preds.groupby('user_id').cumcount() + 1 
    test_preds['item_name'] = test_preds['item_id'].map(item_name_mapper)
    
    
    ''' create 0/1 as indication of interaction:
         > positive event -- 1, if watch_pct is not null
         > negative event -- 0 otherwise
    '''
    positive_preds = pd.merge(test_preds, local_test, how = 'inner', on = ['user_id', 'item_id'])
    positive_preds['target'] = 1
    
    negative_preds = pd.merge(test_preds, local_test, how = 'left', on = ['user_id', 'item_id'])
    negative_preds = negative_preds.loc[negative_preds['watched_pct'].isnull()].sample(frac = .2)
    negative_preds['target'] = 0
    
    # random split to train ranker
    train_users, test_users = train_test_split(
        local_test['user_id'].unique(),
        test_size = .2,
        random_state = 13
        )
    
    cbm_train_set = shuffle(
        pd.concat(
        [positive_preds.loc[positive_preds['user_id'].isin(train_users)],
        negative_preds.loc[negative_preds['user_id'].isin(train_users)]]
        )
    )
    
    cbm_test_set = shuffle(
        pd.concat(
        [positive_preds.loc[positive_preds['user_id'].isin(test_users)],
        negative_preds.loc[negative_preds['user_id'].isin(test_users)]]
        )
    )
    
    # joins user features
    cbm_train_set = pd.merge(cbm_train_set, users_data[['user_id'] + settings.USER_FEATURES],
                            how = 'left', on = ['user_id'])
    cbm_test_set = pd.merge(cbm_test_set, users_data[['user_id'] + settings.USER_FEATURES],
                            how = 'left', on = ['user_id'])
    
    # joins item features
    cbm_train_set = pd.merge(cbm_train_set, movies_metadata[['item_id'] + settings.ITEM_FEATURES],
                            how = 'left', on = ['item_id'])
    cbm_test_set = pd.merge(cbm_test_set, movies_metadata[['item_id'] + settings.ITEM_FEATURES],
                            how = 'left', on = ['item_id'])
    
    x_train, y_train = cbm_train_set.drop(settings.ID_COLS + settings.DROP_COLS + settings.TARGET, axis = 1), cbm_train_set[settings.TARGET]
    x_test, y_test = cbm_test_set.drop(settings.ID_COLS + settings.DROP_COLS + settings.TARGET, axis = 1), cbm_test_set[settings.TARGET]
    
    x_train = x_train.fillna(x_train.mode().iloc[0])
    x_test = x_test.fillna(x_test.mode().iloc[0])
    
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
    
    paths_config = {
        "interactions_data": "artefacts\data\interactions_df.csv",
        "users_data": "artefacts\data\items.csv",
        "items_data": "artefacts\data\items.csv",
    }
    users_data = pd.read_csv(paths_config["users_data"])
    return users_data.loc[users_data['user_id'] == user_id][user_cols].to_dict('records')[0]


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