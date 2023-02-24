import pandas as pd
import os
import json

def save_data(df, fp):

    df.to_csv(fp)

    return

def load_data(fp):

    data = pd.read_csv(fp)

    return data

def current_file_dir():
    
    abspath = os.path.abspath(__file__)
    dir_name = os.path.join(os.path.dirname(abspath), '..')
    
    return dir_name

def load_config_file(filepath):

    with open(filepath) as f:
        config = json.load(f)

    return config

def main():
    return