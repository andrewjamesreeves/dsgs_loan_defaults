import data_utils as du
from pandas.api.types import is_string_dtype

import pandas as pd
import numpy as np
import os

def format_column_names(df, ref):

    df.rename(columns=dict(zip(ref.variable_name_original, ref.variable_name)), inplace=True)

    return df

def convert_categorical_to_numeric(df, ref):

    for col in df.loc[:,(ref[ref.variable_type=='cat']['variable_name'])]:
        df[col] = df[col].astype('category').cat.codes
    
    return df

def split_categorical_variables(df, ref):

    for col in df.loc[:,(ref[ref.variable_type=='cat']['variable_name'])]:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col, prefix_sep='_')], axis=1)

    return df

def convert_binary_string_to_numeric(df, ref):

    for col in df.loc[:,(ref[(ref.variable_type=='binary') & (ref.is_string=='Y')]['variable_name'])]:
        df[col] = df[col].astype('category').cat.codes
    
    return df

def convert_string_elements(df):

    for col in df:
        if is_string_dtype(df[col]) == True:
            df[col] = pd.to_numeric(df[col].str.replace('\D', ''))

    return df


def main(df, data_ref, paths):

    df = (df.pipe(format_column_names, data_ref)
        .pipe(split_categorical_variables, data_ref)
        .pipe(convert_categorical_to_numeric, data_ref)
        .pipe(convert_binary_string_to_numeric, data_ref)
        .pipe(convert_string_elements))
    
    du.save_data(df, os.path.join(os.getcwd(), paths['filepaths']['preprocessed_data'], 'preprocessed_data.csv'))

    return df