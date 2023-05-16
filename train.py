import pandas as pd

from models.lfm import LFMModel
from models.ranker import Ranker
from utils.utils import read_parquet_from_gdrive, clean_interactions, prepare_lightfm_data, item_name_mapper
from data_prep.prepare_ranker_data import prepare_data_for_train
from fire import Fire
import logging
from configs.config import settings

logging.basicConfig(level=logging.INFO)

def train_lfm(data_path: str = None) -> None:
    """
    trains model for a given data with interactions
    :data_path: str, path to parquet with interactions
    """
    interactions_path = settings.RANKER_DATA.INTERACTIONS_DATA_PATH
    if "https" in interactions_path:
        interactions = read_parquet_from_gdrive(interactions_path)
    else:
        interactions = pd.read_parquet(interactions_path)
        
    movie_data_path = settings.RANKER_DATA.MOVIESMETA_DATA_PATH
    if "https" in movie_data_path:
        movies = read_parquet_from_gdrive(movie_data_path)
    else:
        movies = pd.read_parquet(movie_data_path)
        
    clean_inter, clean_movies = clean_interactions(interactions, movies)
    prepare_lightfm_data(clean_inter)
    
    data = pd.read_parquet("artefacts\data\local_train.parquet")

    logging.info('Started training LightFM model...')
    lfm = LFMModel(is_infer = False) # train mode
    lfm.fit(
        data,
        user_col='user_id',
        item_col='movie_id'
    )
    
    item_name_mapper()
    logging.info('Finished training LightFM model!')

def train_ranker():
    """
    executes training pipeline for 2nd level model
    all params are stored in configs
    """

    X_train, X_test, y_train, y_test = prepare_data_for_train()
    logging.info('ranker 1')
    ranker = Ranker(is_infer = False) # train mode
    logging.info('ranker 2')
    ranker.fit(X_train, y_train, X_test, y_test)
    logging.info('Finished training Ranker model!')

if __name__ == '__main__':
    Fire(
    {
        'train_lfm': train_lfm,
        'train_cbm': train_ranker
        }
    )