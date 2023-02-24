import utilities.data_utils as du
from pandas.api.types import is_string_dtype

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler


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

def transform_numeric_variables(df, ref):
    
    # Select only the numerical data.
    numerical_data = df.loc[:,(ref[ref.variable_type.isin(['num'])]['variable_name'])]

    # Create the encoder object
    robust_scaler = RobustScaler()

    # Fit and transform the data with the roust scaler in a new data frame.
    scaled_data = pd.DataFrame(robust_scaler.fit_transform(X=numerical_data), 
                               columns=numerical_data.columns.tolist())
    
    # replace unscaled data with scaled data
    df.drop(columns = numerical_data.columns.tolist(), inplace=True)
    df_scaled = pd.merge(df, scaled_data, left_index = True, right_index=True)

    return df_scaled



def main(df, data_ref, paths, dir_name):

    df = (df.pipe(format_column_names, data_ref)
        .pipe(split_categorical_variables, data_ref)
        .pipe(convert_categorical_to_numeric, data_ref)
        .pipe(convert_binary_string_to_numeric, data_ref)
        .pipe(convert_string_elements))

    #apply encoding


    #apply scalar transformation
    df = transform_numeric_variables(df, data_ref)

    
    du.save_data(df, os.path.join(dir_name, paths['filepaths']['preprocessed_data'], 'preprocessed_data.csv'))

    return df