# Here, you run both models training pipeline using modules we created
# LFM - load wathch data and run fit() method
# Ranker - load candidates based data with features and run fit() method
# REMINDER: it must be active and working. Before that, you shoul finalize prepare_ranker_data.py



from models.lfm import LFMModel
from models.ranker import Ranker
from data_prep.prepare_ranker_data import prepare_data_for_train


lfm = LFMModel()

lfm.fit(
    ...
)

ranker = Ranker()
x_train, y_train, x_test, y_test = prepare_data_for_train(data??)
ranker.fit(x_train, y_train, x_test, y_test, ranker_params={}, categorical_cols=[])