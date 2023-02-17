import pandas as pd

def save_data(df, fp):

    df.to_csv(fp)

    return

def load_data(fp):

    data = pd.read_csv(fp)

    return data

def main():

    return