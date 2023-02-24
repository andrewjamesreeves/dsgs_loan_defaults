import pandas as pd
import os

def save_data(df, fp):

    df.to_csv(fp)

    return

def load_data(fp):

    data = pd.read_csv(fp)

    return data

def main():

    return

def current_file_dir():
    
    abspath = os.path.abspath(__file__)
    dir_name = os.path.dirname(abspath)
    
    return dir_name