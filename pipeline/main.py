import utilities.data_utils as du
import preprocessing.transform_data as td
import models.model_control as mc

import pandas as pd
import json
import os


def main():
    
    #get file path of current file
    dir_name = du.current_file_dir()

    #load config files
    paths = du.load_config_file(os.path.join(dir_name, 'configuration_files\\filepaths.json'))
    models_config = du.load_config_file(os.path.join(dir_name, 'configuration_files\\models_config.json'))

    #load training_data and data reference
    training_data = du.load_data(os.path.join(dir_name, paths['filepaths']['training_data']))
    reference_data = du.load_data(os.path.join(dir_name, paths['filepaths']['reference_data']))
    test_data = du.load_data(os.path.join(dir_name, paths['filepaths']['test_data']))

    #pass data to transform data/preprocessing data areas
    training_data_pp, test_data_pp = td.main(training_data, reference_data, paths, dir_name, split=True)

    #pass data into models and combine evaluation metrics
    mc.main(training_data_pp, test_data_pp, reference_data, models_config, paths, dir_name)
    
    # prepcrocess test data
    test_data = td.main(test_data, reference_data, paths, dir_name, split=False)


if __name__ == "__main__":
  main()