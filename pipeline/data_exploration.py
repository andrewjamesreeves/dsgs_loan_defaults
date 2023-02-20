import data_utils as du
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def distribution_graph(df, column):
    fig, ax = plt.subplots()    
    ax = df[column].plot.density(title="Distribution of " +  column, label=column)

    mean_value= df[column].mean()
    
    min_value = df[column].min()
    max_value = df[column].max()
    textstr = '\n'.join((
        r'$\min=%.2f$' % (min_value, ),
        r'$\max=%.2f$' % (max_value, )))
    
    # define matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

    # Plotting the mean value on the distribution against the density.
    plt.axvline(x=mean_value, color='r', linestyle='--', label="mean")
    ax.legend()
    plt.savefig(os.path.join( du.current_file_dir(), paths['filepaths']['output'] , "distribution/" + column + ".png"))


#load config files
paths_file = os.path.join(du.current_file_dir(), 'configuration_files\\filepaths.json')
with open(paths_file) as f:
    paths = json.load(f)

# load in processed data
preprocessed_data = du.load_data(os.path.join(du.current_file_dir(), paths['filepaths']['preprocessed_data'], "preprocessed_data.csv"))

# get list of numerical variables
ref_file = du.load_data(os.path.join(du.current_file_dir(), paths['filepaths']['reference_data']))

# remove accounts_delinquent as only value is 0 and this breaks the loop
ref_file_filtered = ref_file.loc[(ref_file["variable_name"] != "accounts_delinquent")]

#produce distribution graphs 
for col in ref_file_filtered[ref_file_filtered.variable_type.isin(['con','dis', 'discrete'])]['variable_name']:
        distribution_graph(preprocessed_data, col)

