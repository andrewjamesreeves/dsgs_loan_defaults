from models.logit import apply_logit
from models.lasso import apply_lasso

def get_available_models():

    models = {
        "logit": apply_logit,
        "lasso": apply_lasso
    }

    return models

def run_models(models_config, models, training_data, test_data):

    for model in models_config:
        print(f'Running Model -> {model}')
        model_output = models[model](training_data, test_data, models_config[model])

    return

def main(training_data, test_data, reference_data, models_config, paths, dir_name):

    models = get_available_models()

    run_models(models_config, models, training_data, test_data)


    return