# from dynaconf import Dynaconf
# import logging

# logger = logging
# logging.basicConfig(level=logging.INFO)

# settings = Dynaconf(
#     settings_files=[
#         "configs/user_features.toml",
#     ]
# )
from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=[
        "configs/features.toml",
        "configs/ranker_data_prep.toml",
        "configs/models_params.toml"
    ]
)