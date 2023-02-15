import preprocessing.load_data as ld
import preprocessing.transform_data as td

import pandas as pd
import json
import os


def main():

    #load config files
    paths_file = os.path.join(os.getcwd(), 'configuration_files\\filepaths.json')
    with open(paths_file) as f:
        paths = json.load(f)

    #set up load data
    training_data = ld.main(os.path.join(os.getcwd(), paths['filepaths']['training_data']))

    #pass data to transform data/preprocessing data areas
    td.main(training_data)


    return

if __name__ == "__main__":
  main()