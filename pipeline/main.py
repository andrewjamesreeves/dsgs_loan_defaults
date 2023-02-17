import data_utils as du
import preprocessing.transform_data as td

import pandas as pd
import json
import os


def main():

    #load config files
    paths_file = os.path.join(os.getcwd(), 'configuration_files\\filepaths.json')
    with open(paths_file) as f:
        paths = json.load(f)

    #load training_data and data reference
    training_data = du.load_data(os.path.join(os.getcwd(), paths['filepaths']['training_data']))
    reference_data = du.load_data(os.path.join(os.getcwd(), paths['filepaths']['reference_data']))

    #pass data to transform data/preprocessing data areas
    training_data_pp = td.main(training_data, reference_data, paths)

    return

if __name__ == "__main__":
  main()