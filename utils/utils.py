import json
from datetime import datetime, timedelta
import pickle
import datetime as dt

import dill
import numpy as np
import pandas as pd
from cachetools import TTLCache, cached


@cached(cache=TTLCache(maxsize=1024, ttl=timedelta(hours=12), timer=datetime.now))
def read_parquet_from_gdrive(url):
    """
    gets csv data from a given url (from file -> share -> copy link)
    :url: *****/view?usp=share_link
    """
    file_id = url.split("/")[-2]
    file_path = "https://drive.google.com/uc?export=download&id=" + file_id
    data = pd.read_parquet(file_path)

    return data


def generate_lightfm_recs_mapper(
    model: object,
    item_ids: list,
    known_items: dict,
    user_features: list,
    item_features: list,
    N: int,
    user_mapping: dict,
    item_inv_mapping: dict,
    num_threads: int = 4,
):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.predict(
            user_id,
            item_ids,
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
        )

        additional_N = len(known_items[user_id]) if user_id in known_items else 0
        total_N = N + additional_N
        top_cols = np.argpartition(recs, -np.arange(total_N))[-total_N:][::-1]

        final_recs = [item_inv_mapping[item] for item in top_cols]
        if additional_N > 0:
            filter_items = known_items[user_id]
            final_recs = [item for item in final_recs if item not in filter_items]
        return final_recs[:N]

    return _recs_mapper


def save_model(model: object, path: str):
    with open(f"{path}", "wb") as obj_path:
        dill.dump(model, obj_path)


def load_model(path: str):
    with open(path, "rb") as obj_file:
        obj = dill.load(obj_file)
    return obj


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)

def item_name_mapper():
    movies_metadata = pd.read_parquet("artefacts\data\clean_movies.parquet")
    
    res = dict(zip(movies_metadata['movie_id'], movies_metadata['title']))
    with open('artefacts\item_name_mapper_data.pkl', 'wb') as fp:
        pickle.dump(res, fp)
        
def clean_interactions(interactions, movies):
    df = movies.join(interactions.set_index('movie_id'), on='movie_id', how='left')
    df['watch_duration_minutes'] = df.apply(lambda x: x['watch_duration_minutes']
                                        if (x['watch_duration_minutes'] < x['duration']
                                        or pd.isna(x['duration']))
                                        else x['duration'], axis=1)
    
    avg_watch = df.groupby('movie_id')['watch_duration_minutes'].mean().to_dict()
    median_watch = df.groupby('movie_id')['watch_duration_minutes'].median().to_dict()
    popularity = df.groupby('movie_id')['user_id'].nunique().to_dict()
    rewatches = (df.groupby('movie_id')['user_id'].count() / df.groupby('movie_id')['user_id'].nunique()).to_dict()
    
    check_avg_const = movies[movies['duration'].notna()]
    check_avg_const['avg_watch'] = check_avg_const['movie_id'].apply(lambda x: avg_watch[x])
    check_avg_const = check_avg_const[check_avg_const['avg_watch'].notna()]
    
    movies['duration'] = movies['duration'].fillna(0)
    duration_dict = movies.set_index('movie_id')['duration'].to_dict()

    movies['avg_watch'] = movies['movie_id'].apply(lambda x: avg_watch[x] if pd.notna(avg_watch[x]) else duration_dict[x]*0.96)
    movies['median_watch'] = movies['movie_id'].apply(lambda x: median_watch[x] if pd.notna(median_watch[x]) else duration_dict[x])
    movies['popularity'] = movies['movie_id'].apply(lambda x: popularity[x] if pd.notna(popularity[x]) else 0)
    movies['rewatches'] = movies['movie_id'].apply(lambda x: rewatches[x] if pd.notna(rewatches[x]) else 0)
    
    def f(dur, avg):
        if dur == 0:
            return avg/0.96
        else:
            return dur
    movies['duration_est'] = movies.apply(lambda x: f(x['duration'], x['avg_watch']), axis=1)
    
    movies['avg_watch'] = movies['avg_watch'] * movies['popularity'].apply(lambda x: min(1, x))
    movies['median_watch'] = movies['median_watch'] * movies['popularity'].apply(lambda x: min(1, x))
    movies['watched_ratio'] = (movies['avg_watch']/movies['duration_est']).fillna(0)
    
    movies['age_rating'] = movies['age_rating'].fillna(movies['age_rating'].median())
    movies['release_world'] = movies['release_world'].fillna('0-0-0')
    movies = movies.fillna('["UNKNOWN"]')
    
    movies['main_genre'] = movies['genres'].apply(lambda x: x.strip('[]').split(',')[0].strip('"'))
    movies['main_actor'] = movies['actors'].apply(lambda x: x.strip('[]').split(',')[0].strip('"'))
    movies['main_director'] = movies['director'].apply(lambda x: x.strip('[]').split(',')[0].strip('"'))
    movies['main_country'] = movies['country'].apply(lambda x: x.strip('[]').split(',')[0].strip('"'))

    movies['released_year'] =  movies['release_world'].apply(lambda x: x.strip('\'').split('-')[0])
    movies['released_month'] =  movies['release_world'].apply(lambda x: x.strip('\'').split('-')[1])
    
    clean_movies = movies.drop(columns=['genres', 'actors', 'director', 'country', 'release_world'])
    
    ## interaction data
    df = clean_movies.join(interactions.set_index('movie_id'), on='movie_id', how='inner')
    actors = df.groupby('user_id')['main_actor'].agg(pd.Series.mode).to_dict()
    directors = df.groupby('user_id')['main_director'].agg(pd.Series.mode).to_dict()
    countries = df.groupby('user_id')['main_country'].agg(pd.Series.mode).to_dict()
    genres = df.groupby('user_id')['main_genre'].agg(pd.Series.mode).to_dict()
    
    df['fav_actor'] = df['user_id'].apply(lambda x: actors[x][0] if type(actors[x])==np.ndarray else actors[x])
    df['fav_director'] = df['user_id'].apply(lambda x: directors[x][0] if type(directors[x])==np.ndarray else directors[x])
    df['fav_country'] = df['user_id'].apply(lambda x: countries[x][0] if type(countries[x])==np.ndarray else countries[x])
    df['fav_genre'] = df['user_id'].apply(lambda x: genres[x][0] if type(genres[x])==np.ndarray else genres[x])
    
    df['watch_duration_minutes'] = df.apply(lambda x: x['watch_duration_minutes']
                                        if x['watch_duration_minutes'] < x['duration_est']
                                        else x['duration_est'], axis=1)
    
    df['watched_ratio'] = df['watch_duration_minutes']/df['duration_est']
    clean_inter = df[['movie_id', 'user_id', 'year', 'month', 'day', 'watch_duration_minutes', 
                  'fav_actor', 'fav_director', 'fav_country', 'fav_genre', 'watched_ratio']].reset_index(drop=True)
    
    clean_inter.to_parquet('artefacts\data\clean_inter.parquet')
    clean_movies.to_parquet('artefacts\data\clean_movies.parquet')
    
    return clean_inter, clean_movies
    

def prepare_lightfm_data(clean_inter):
    interactions_filtered = clean_inter
    
    date = []
    for index, row in interactions_filtered.iterrows():
        date.append(pd.Timestamp(year=row['year'],
                month=row['month'],
                day= row['day']))
        
    interactions_filtered['last_watch_dt'] = pd.Series(date)
    interactions_filtered['last_watch_dt'] = pd.to_datetime(interactions_filtered['last_watch_dt'])
    
    # set dates params for filter
    MAX_DATE = interactions_filtered['last_watch_dt'].max()
    MIN_DATE = interactions_filtered['last_watch_dt'].min()
    TEST_INTERVAL_DAYS = 7
    
    TEST_MAX_DATE = MAX_DATE - dt.timedelta(days = TEST_INTERVAL_DAYS)
    global_train = interactions_filtered.loc[interactions_filtered['last_watch_dt'] < TEST_MAX_DATE]
    global_test = interactions_filtered.loc[interactions_filtered['last_watch_dt'] >= TEST_MAX_DATE]
    
    seen_global_train = global_train.groupby('user_id')['movie_id'].unique().to_dict()
    seen_global_test = global_test.groupby('user_id')['movie_id'].unique().to_dict()
    
    # now, we define "local" train and test to use some part of the global train for ranker
    local_train_thresh = global_train['last_watch_dt'].quantile(q = .7, interpolation = 'nearest')

    local_train = global_train.loc[global_train['last_watch_dt'] < local_train_thresh]
    local_test = global_train.loc[global_train['last_watch_dt'] >= local_train_thresh]
    
    local_test = local_test.loc[local_test['user_id'].isin(local_train['user_id'].unique())]
    
    seen_local_train = local_train.groupby('user_id')['movie_id'].unique().to_dict()
    seen_local_test = local_test.groupby('user_id')['movie_id'].unique().to_dict()
    
    with open('artefacts\data\seen_local_train.pkl', 'wb') as fp:
        pickle.dump(seen_local_train, fp)
    local_train.to_parquet('artefacts\data\local_train.parquet')
    local_test.to_parquet('artefacts\data\local_test.parquet')