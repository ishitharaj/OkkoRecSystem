from dynaconf import Dynaconf
import logging

logger = logging
logging.basicConfig(level=logging.INFO)

settings = Dynaconf(
    settings_files=[
        "configs/user_features.toml",
    ]
)