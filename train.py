# Here, you run both models training pipeline using modules we created
# LFM - load wathch data and run fit() method
# Ranker - load candidates based data with features and run fit() method
# REMINDER: it must be active and working. Before that, you shoul finalize prepare_ranker_data.py
import pandas as pd
from configs.config import settings
from models.lfm import LFMModel
from models.ranker import Ranker
from data_prep.prepare_ranker_data import prepare_data_for_train
from utils.utils import prepare_interaction_data

paths_config = {
        "interactions_data": "artefacts/data/interactions_df.csv",
        "users_data": "artefacts/data/users.csv",
        "items_data": "artefacts/data/items.csv"
        }

interactions = pd.read_csv(paths_config["interactions_data"])
movies_metadata = pd.read_csv(paths_config["items_data"])

lfm = LFMModel()
ranker = Ranker()

global_train, global_test, local_train, local_test = prepare_interaction_data(paths_config)
lfm.fit(
    local_train, # used prepated data rathar than pure interactions, need review
    'user_id',
    'item_id',
    # change parameters as needed
    {
        "epochs":1,
        "no_components":10,
        "learning_rate":0.05,
        "loss":"logistic",
        "max_sampled":10,
        "random_state":42,
    }
)

x_train, y_train, x_test, y_test = prepare_data_for_train(paths_config, lfm.lfm, global_train, global_test, local_train, local_test)
ranker.fit(
    x_train, 
    y_train, 
    x_test, 
    y_test, 
    {
        "loss_function": "CrossEntropy",
        "iterations":5000,
        "learning_rate":.1,
        "depth":6,
        "random_state":1234,
        "verbose":True,
    },
    settings.CATEGORICAL_COLS
    )