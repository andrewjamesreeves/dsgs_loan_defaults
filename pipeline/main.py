import data_utils as du
import preprocessing.transform_data as td

import pandas as pd
import json
import os


def main():
    
    #get file path of current file
    dir_name = du.current_file_dir()

    #load config files
    paths_file = os.path.join(dir_name, 'configuration_files\\filepaths.json')
    with open(paths_file) as f:
        paths = json.load(f)

    #load training_data and data reference
    training_data = du.load_data(os.path.join(dir_name, paths['filepaths']['training_data']))
    reference_data = du.load_data(os.path.join(dir_name, paths['filepaths']['reference_data']))

    #pass data to transform data/preprocessing data areas
    training_data_pp = td.main(training_data, reference_data, paths, dir_name)

    # return

if __name__ == "__main__":
  main()