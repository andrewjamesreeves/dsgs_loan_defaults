# from models.logit import apply_logit
# from models.lasso import apply_lasso
# from models.random_forest import apply_rfc
from models.model_functions import apply_logit, apply_rfc, apply_lasso
from sklearn.metrics import precision_recall_fscore_support
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

def evaluation_metrics_df(Y_test, yhat, config):
    # produce weighted metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(Y_test, yhat, 
                                                          average='weighted')
    
    # produce macro metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(Y_test, yhat, 
                                                          average='macro')
    
    output = pd.DataFrame({'model': [config["model"]],
                           'precision_macro':[precision_macro],
                           'recall_macro':[recall_macro],
                           'f1_macro': [f1_macro],
                           'precision_weighted': [precision_weighted],
                           'recall_weighted': [recall_weighted],
                           'f1_weighted': [f1_weighted],
                           })
    return output

def run_models_and_combine_metrics(models_config, models, training_data, test_data, paths, dir_name, submission_data = None):
    
    metrics = pd.DataFrame()
    metrics_submission = pd.DataFrame()
    
    for model in models_config:
        print(f'Running Model -> {model}')
        Y_test, yhat, chosen_model, variable_selection_columns = models[models_config[model]['model']](training_data, test_data, models_config[model])
        
        model_output = evaluation_metrics_df(Y_test, yhat, models_config[model])
        # obtain metrics
        metrics = metrics.append(model_output)
        
        if model != "lasso" and submission_data is not None:
            X_test_submission = submission_data[variable_selection_columns]
            yhat_submission = chosen_model.predict(X_test_submission)
            submission_output = pd.DataFrame(data = {"ID" : submission_data["ID"],
                                                        "Loan Status" : yhat_submission})

            # save output  
            du.save_data(submission_output, os.path.join(dir_name, paths['filepaths']['results'], f'submission_output_{model}.csv'))
    
    
    return metrics


def main(training_data, test_data, reference_data, models_config, paths, dir_name, submission_data = None):

    models = get_available_models()
    
    if submission_data is not None:
            model_comparison  = run_models_and_combine_metrics(models_config, models, training_data, test_data, paths, dir_name, submission_data)
    
    else:
            model_comparison = run_models_and_combine_metrics(models_config, models, training_data, paths, dir_name, test_data)
            
    # save table  
    du.save_data(model_comparison, os.path.join(dir_name, paths['filepaths']['results'], 'model_comparison.csv'))
    
    return