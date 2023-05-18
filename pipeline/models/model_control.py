# from models.logit import apply_logit
# from models.lasso import apply_lasso
# from models.random_forest import apply_rfc
from models.model_functions import apply_logit, apply_rfc, apply_lasso
import pandas as pd
import utilities.data_utils as du
import os


def get_available_models():

    models = {
        "logit": apply_logit,
        "lasso": apply_lasso,
        "random_forest" : apply_rfc
    }

    return models

def run_models_and_combine_metrics(models_config, models, training_data, test_data):
    
    metrics = pd.DataFrame()
    
    for model in models_config:
        print(f'Running Model -> {model}')
        model_output = models[models_config[model]['model']](training_data, test_data, {**models_config[model], "model_name":str(model)})
        
        # obtain metrics
        metrics = metrics.append(model_output)

    return metrics


def main(training_data, test_data, reference_data, models_config, paths, dir_name):

    models = get_available_models()

    model_comparison = run_models_and_combine_metrics(models_config, models, training_data, test_data)
    
    # save table  
    du.save_data(model_comparison, os.path.join(dir_name, paths['filepaths']['results'], 'model_comparison.csv'))

    return