import pandas as pd

def format_column_names(df, ref):

    df.rename(columns=dict(zip(ref.variable_name_original, ref.variable_name)), inplace=True)

    return df

def convert_categorical_to_numeric(df, ref):

    for col in df.loc[:,(ref[ref.variable_type=='cat']['variable_name'])]:
        df[col] = df[col].astype('category').cat.codes
    
    return df

def convert_binary_string_to_numeric(df, ref):

    for col in df.loc[:,(ref[(ref.variable_type=='binary') & (ref.is_string==str(1))]['variable_name'])]:
        df[col] = df[col].astype('category').cat.codes
    
    return df


def main(df, data_ref):

    (df.pipe(format_column_names, data_ref)
        .pipe(convert_categorical_to_numeric, data_ref)
        .pipe(convert_binary_string_to_numeric, data_ref))
    

    return