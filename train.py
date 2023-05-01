# # Here, you run both models training pipeline using modules we created
# # LFM - load wathch data and run fit() method
# # Ranker - load candidates based data with features and run fit() method
# # REMINDER: it must be active and working. Before that, you shoul finalize prepare_ranker_data.py
# import pandas as pd
# from configs.config import settings
# from models.lfm import LFMModel
# from models.ranker import Ranker
# from data_prep.prepare_ranker_data import prepare_data_for_train
# from utils.utils import prepare_interaction_data

# paths_config = {
#         "interactions_data": "artefacts/data/interactions_df.csv",
#         "users_data": "artefacts/data/users.csv",
#         "items_data": "artefacts/data/items.csv"
#         }

# interactions = pd.read_csv(paths_config["interactions_data"])
# movies_metadata = pd.read_csv(paths_config["items_data"])

# lfm = LFMModel()
# ranker = Ranker()

# global_train, global_test, local_train, local_test = prepare_interaction_data(paths_config)
# lfm.fit(
#     interactions,
#     'user_id',
#     'item_id',
#     # change parameters as needed
#     {
#         "epochs":1,
#         "no_components":10,
#         "learning_rate":0.05,
#         "loss":"logistic",
#         "max_sampled":10,
#         "random_state":42,
#     }
# )

# x_train, y_train, x_test, y_test = prepare_data_for_train(paths_config, lfm.lfm, global_train, global_test, local_train, local_test)
# ranker.fit(
#     x_train, 
#     y_train, 
#     x_test, 
#     y_test, 
#     {
#         "loss_function": "CrossEntropy",
#         "iterations":5000,
#         "learning_rate":.1,
#         "depth":6,
#         "random_state":1234,
#         "verbose":True,
#     },
#     settings.CATEGORICAL_COLS
#     )
import pandas as pd

from models.lfm import LFMModel
from models.ranker import Ranker
from utils.utils import read_parquet_from_gdrive
from data_prep.prepare_ranker_data import prepare_data_for_train

from fire import Fire

import logging

def train_lfm(data_path: str = None) -> None:
    """
    trains model for a given data with interactions
    :data_path: str, path to parquet with interactions
    """
    if data_path is None:
        logging.warning('Local data path is not set... Using default from GDrive')
        data = read_parquet_from_gdrive('https://drive.google.com/file/d/1MomVjEwY2tPJ845zuHeTPt1l53GX2UKd/view?usp=share_link')

    else:
        logging.info(f'Reading data from local path: {data_path}')
        data = pd.read_parquet(data_path)

    logging.info('Started training LightFM model...')
    lfm = LFMModel(is_infer = False) # train mode
    lfm.fit(
        data,
        user_col='user_id',
        item_col='item_id'
    )
    logging.info('Finished training LightFM model!')

def train_ranker():
    """
    executes training pipeline for 2nd level model
    all params are stored in configs
    """

    X_train, X_test, y_train, y_test = prepare_data_for_train()
    ranker = Ranker(is_infer = False) # train mode
    ranker.fit(X_train, y_train, X_test, y_test)
    logging.info('Finished training Ranker model!')

if __name__ == '__main__':
    Fire(
    {
        'train_lfm': train_lfm,
        'train_cbm': train_ranker
        }
    )