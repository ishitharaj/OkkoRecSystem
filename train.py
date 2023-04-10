# Here, you run both models training pipeline using modules we created
# LFM - load wathch data and run fit() method
# Ranker - load candidates based data with features and run fit() method
# REMINDER: it must be active and working. Before that, you shoul finalize prepare_ranker_data.py
import pandas as pd
from configs.config import settings
from models.lfm import LFMModel
from models.ranker import Ranker
from data_prep.prepare_ranker_data import prepare_data_for_train

paths_config = {
        "interactions_data": "artefacts\data\interactions_df.csv",
        "users_data": "artefacts\data\items.csv",
        "items_data": "artefacts\data\items.csv",
    }

interactions = pd.read_csv(paths_config["interactions_data"])

lfm = LFMModel()
ranker = Ranker()

lfm.fit(
    interactions,
    'user_id',
    'item_id',
    # change parameters as needed
    {
        "epochs":10,
        "no_components":10,
        "learning_rate":0.05,
        "loss":"logistic",
        "max_sampled":10,
        "random_state":42,
    }
)

x_train, y_train, x_test, y_test= prepare_data_for_train(paths_config)
ranker.fit(x_train, y_train, x_test, y_test)