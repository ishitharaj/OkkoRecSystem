# OkkoRecSystem

Okko Recommendation System Project

# Full RecSys Pipeline
Here we have the full pipeline to train and make inference using two-level model architecture

![image](https://user-images.githubusercontent.com/38528963/230792021-0e406ed5-6fe7-4177-ac20-52881d869864.png)


## Repo Structure
- /artefacts - local storage for models artefacts;
- /data_prep - data preparation modules to be used during training_pipeline;
- /models - model fit and inference pipeline
- /utils - some common functions thatn can be used everywhere

## DEV Guide  

### To install dependencies from poetry:
1. ``` pip install poetry ```  
2. ``` poetry config virtualenvs.in-project true```  
3. ``` poetry install```  
4.  Run poetry run ```pip install lightfm==1.17 --no-use-pep517``` to workaround install lfm model issue  
5. ``` poetry run python train.py train_lfm ``` runs training pipeline for candidates model (run within created env)  
6. ``` poetry run python train.py train_cbm ``` runs ranker training pipeline (takes a while)  
  
### To deploy model
```./start.sh ``` - To run containers  
```./reset.sh``` - To remove containers  

### Team Members:  
  
Ishitha Rajapakse  
Elena Tkachenko @el-tka  
Nikita Markin @  
Victoria Matskovskaya @viktoriyams 


